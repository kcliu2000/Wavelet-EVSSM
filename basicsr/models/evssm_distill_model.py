import torch
import torch.nn.functional as F
from collections import OrderedDict
from basicsr.models.image_restoration_model import ImageRestorationModel
from basicsr.models.archs.EVSSM_arch import EVSSM

# ==========================================================
# 📌 武器一：小波引導知識蒸餾 Loss (Wavelet KD Loss)
# ==========================================================
class HaarWaveletKDLoss(torch.nn.Module):
    def __init__(self, high_freq_weight=2.0):
        super(HaarWaveletKDLoss, self).__init__()
        self.hf_weight = high_freq_weight
        h0 = torch.tensor([[1/2, 1/2], [1/2, 1/2]]).view(1, 1, 2, 2)
        h1 = torch.tensor([[1/2, 1/2], [-1/2, -1/2]]).view(1, 1, 2, 2)
        h2 = torch.tensor([[1/2, -1/2], [1/2, -1/2]]).view(1, 1, 2, 2)
        h3 = torch.tensor([[1/2, -1/2], [-1/2, 1/2]]).view(1, 1, 2, 2)
        filters = torch.cat([h0, h1, h2, h3], dim=0).repeat(3, 1, 1, 1)
        self.register_buffer('filters', filters)

    def forward(self, student_img, teacher_img):
        stu_dwt = F.conv2d(student_img, self.filters, stride=2, groups=3)
        tea_dwt = F.conv2d(teacher_img, self.filters, stride=2, groups=3)
        diff = torch.abs(stu_dwt - tea_dwt)
        weights = torch.ones_like(diff) * self.hf_weight
        weights[:, [0, 4, 8], :, :] = 1.0 
        return torch.mean(diff * weights)

# ==========================================================
# 📌 武器二：Mamba 潛在空間竊聽器 (Feature Extractor)
# ==========================================================
class FeatureExtractor:
    def __init__(self, model, target_layers):
        self.features = {}
        self.hooks = []
        for name, module in model.named_modules():
            if name in target_layers:
                self.hooks.append(module.register_forward_hook(self.get_hook(name)))
                
    def get_hook(self, layer_name):
        def hook(module, input, output):
            # 處理 SSM 可能回傳 tuple 的情況
            if isinstance(output, tuple):
                self.features[layer_name] = output[0]
            else:
                self.features[layer_name] = output
        return hook
        
    def clear(self):
        self.features.clear()

# ==========================================================

def build_network_local(opt):
    opt = opt.copy()
    net_type = opt.pop('type')
    if net_type == 'EVSSM':
        return EVSSM(**opt)
    else:
        raise NotImplementedError(f'不支援: {net_type}')

class EVSSMDistillationModel(ImageRestorationModel):
    def __init__(self, opt):
        super(EVSSMDistillationModel, self).__init__(opt)
        
        # 1. 載入 Teacher 模型
        self.net_teacher = build_network_local(opt['network_teacher'])
        self.net_teacher = self.net_teacher.to(self.device)
         # 💡 強制指定：不管有沒有 resume，Teacher 永遠讀取最原始的官方權重
        load_path = opt.get('path', {}).get('pretrain_network_teacher', None)
        
        # BasicSR 遇到 resume_state 時會偷偷把路徑改成 .../net_teacher_130000.pth
        # 我們在這裡把它強制改回我們真正在 YAML 裡寫的原始路徑
        actual_teacher_path = "/home/m11302124/EVSSM-Baseline/pretrained_model/net_g_GoPro.pth"
        
        if load_path is not None:
            # 使用我們強制鎖定的 actual_teacher_path
            self.load_network(self.net_teacher, actual_teacher_path, True, 'params')
        else:
            raise ValueError("請在 YAML 設定檔提供 Teacher 模型的預訓練權重路徑！")
        self.net_teacher.eval()
        for param in self.net_teacher.parameters():
            param.requires_grad = False

        self.kd_weight = opt['train'].get('kd_weight', 0.5)

        # 2. 載入 FFTLoss
        self.cri_fft = None
        if opt['train'].get('fft_opt'):
            try:
                from basicsr.models.losses.losses import FFTLoss
                fft_opt = opt['train']['fft_opt'].copy()
                fft_opt.pop('type', None)
                self.cri_fft = FFTLoss(**fft_opt).to(self.device)
            except ImportError:
                pass

        # 🟢 [開關] 讀取 YAML: 是否啟用小波蒸餾
        self.use_wavelet_kd = opt['train'].get('use_wavelet_kd', False)
        if self.use_wavelet_kd:
            self.wav_kd_weight = opt['train'].get('wavelet_kd_weight', 0.5)
            self.wavelet_kd_loss = HaarWaveletKDLoss(high_freq_weight=2.0).to(self.device)

        # 🔴 [開關] 讀取 YAML: 是否啟用特徵蒸餾
        self.use_feature_kd = opt['train'].get('use_feature_kd', False)
        if self.use_feature_kd:
            self.feat_kd_weight = opt['train'].get('feat_kd_weight', 0.1)
            self.target_layers_teacher = opt['train'].get('target_layers_teacher', [])
            self.target_layers_student = opt['train'].get('target_layers_student', [])
            
            if self.target_layers_teacher and self.target_layers_student:
                self.tea_extractor = FeatureExtractor(self.net_teacher, self.target_layers_teacher)
                self.stu_extractor = FeatureExtractor(self.net_g, self.target_layers_student)

    def optimize_parameters(self, current_iter, tb_logger=None):
        self.optimizer_g.zero_grad()
        
        preds_student = self.net_g(self.lq)
        with torch.no_grad():
            preds_teacher = self.net_teacher(self.lq)
            
        l_total = 0
        loss_dict = OrderedDict()
        
        # [基礎 Loss]: Pixel + FFT + Output KD
        if self.cri_pix:
            l_pix = self.cri_pix(preds_student, self.gt)
            l_total += l_pix
            loss_dict['l_pix'] = l_pix
            
        if self.cri_fft:
            l_fft = self.cri_fft(preds_student, self.gt)
            l_total += l_fft
            loss_dict['l_fft'] = l_fft
            
        l_kd = F.l1_loss(preds_student, preds_teacher) * self.kd_weight
        l_total += l_kd
        loss_dict['l_kd'] = l_kd

        # 🟢 [武器一啟動]: 小波引導蒸餾
        if self.use_wavelet_kd:
            l_wav_kd = self.wavelet_kd_loss(preds_student, preds_teacher) * self.wav_kd_weight
            l_total += l_wav_kd
            loss_dict['l_wav_kd'] = l_wav_kd

        # 🔴 [武器二啟動]: Mamba 特徵蒸餾
        if self.use_feature_kd and hasattr(self, 'tea_extractor'):
            l_feat = 0
            for t_name, s_name in zip(self.target_layers_teacher, self.target_layers_student):
                if t_name in self.tea_extractor.features and s_name in self.stu_extractor.features:
                    feat_t = self.tea_extractor.features[t_name]
                    feat_s = self.stu_extractor.features[s_name]
                    l_feat += F.l1_loss(feat_s, feat_t)
            
            if l_feat > 0:
                l_feat = l_feat * self.feat_kd_weight
                l_total += l_feat
                loss_dict['l_feat_kd'] = l_feat
                
            self.tea_extractor.clear()
            self.stu_extractor.clear()

        l_total.backward()
        self.optimizer_g.step()
        self.log_dict = self.reduce_loss_dict(loss_dict)