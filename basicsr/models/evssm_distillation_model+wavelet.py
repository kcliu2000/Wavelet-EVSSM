import torch
import torch.nn.functional as F
from collections import OrderedDict
from basicsr.models.image_restoration_model import ImageRestorationModel
from basicsr.models.archs.EVSSM_arch import EVSSM

# ==========================================================
# 📌 新增：小波引導知識蒸餾 Loss (Wavelet KD Loss)
# 透過 Haar 小波轉換提取高低頻，並強迫學生著重模仿老師的高頻邊緣
# ==========================================================
class HaarWaveletKDLoss(torch.nn.Module):
    def __init__(self, high_freq_weight=2.0):
        super(HaarWaveletKDLoss, self).__init__()
        self.hf_weight = high_freq_weight
        
        # 定義 Haar 小波的 4 個濾波器
        h0 = torch.tensor([[1/2, 1/2], [1/2, 1/2]]).view(1, 1, 2, 2)
        h1 = torch.tensor([[1/2, 1/2], [-1/2, -1/2]]).view(1, 1, 2, 2)
        h2 = torch.tensor([[1/2, -1/2], [1/2, -1/2]]).view(1, 1, 2, 2)
        h3 = torch.tensor([[1/2, -1/2], [-1/2, 1/2]]).view(1, 1, 2, 2)
        
        # 合併濾波器並適應 RGB 3 通道
        filters = torch.cat([h0, h1, h2, h3], dim=0).repeat(3, 1, 1, 1)
        self.register_buffer('filters', filters)

    def forward(self, student_img, teacher_img):
        # 進行 DWT 轉換
        stu_dwt = F.conv2d(student_img, self.filters, stride=2, groups=3)
        tea_dwt = F.conv2d(teacher_img, self.filters, stride=2, groups=3)
        
        # 計算 L1 絕對誤差
        diff = torch.abs(stu_dwt - tea_dwt)
        
        # 設定權重矩陣：低頻(LL)權重為1，高頻(LH,HL,HH)權重放大
        weights = torch.ones_like(diff) * self.hf_weight
        # Channel 0, 4, 8 分別對應 R, G, B 的 LL (低頻)
        weights[:, [0, 4, 8], :, :] = 1.0 
        
        loss = torch.mean(diff * weights)
        return loss

# ==========================================================

def build_network_local(opt):
    opt = opt.copy()
    net_type = opt.pop('type')
    if net_type == 'EVSSM':
        return EVSSM(**opt)
    else:
        raise NotImplementedError(f'目前的蒸餾檔案僅支援 EVSSM 架構，不支援: {net_type}')

class EVSSMDistillationModel(ImageRestorationModel):
    def __init__(self, opt):
        super(EVSSMDistillationModel, self).__init__(opt)
        
        # 1. 建立並載入 Teacher 模型
        self.net_teacher = build_network_local(opt['network_teacher'])
        self.net_teacher = self.net_teacher.to(self.device)
        
        load_path = opt.get('path', {}).get('pretrain_network_teacher', None)
        if load_path is not None:
            self.load_network(self.net_teacher, load_path, True, 'params')
        else:
            raise ValueError("請在 YAML 設定檔提供 Teacher 模型的預訓練權重路徑！")
            
        self.net_teacher.eval()
        for param in self.net_teacher.parameters():
            param.requires_grad = False

        self.kd_weight = opt['train'].get('kd_weight', 0.5)

        # 2. 載入 FFTLoss (手動載入防呆機制)
        self.cri_fft = None
        if opt['train'].get('fft_opt'):
            if hasattr(self, 'cri_fft') and self.cri_fft is not None:
                pass 
            else:
                try:
                    from basicsr.models.losses.losses import FFTLoss
                    fft_opt = opt['train']['fft_opt'].copy()
                    fft_opt.pop('type', None)
                    self.cri_fft = FFTLoss(**fft_opt).to(self.device)
                except ImportError:
                    print("警告: 找不到 FFTLoss，將僅使用 L1 + KD Loss。")

        # 📌 3. 初始化我們自定義的小波蒸餾 Loss
        # 這裡設定 high_freq_weight=2.0，代表高頻誤差的懲罰是低頻的兩倍
        self.wavelet_kd_loss = HaarWaveletKDLoss(high_freq_weight=2.0).to(self.device)

    def optimize_parameters(self, current_iter, tb_logger=None):
        self.optimizer_g.zero_grad()
        
        preds_student = self.net_g(self.lq)
        
        with torch.no_grad():
            preds_teacher = self.net_teacher(self.lq)
            
        l_total = 0
        loss_dict = OrderedDict()
        
        # [Loss 1]: 學生與 GT 之間的 Pixel Loss (L1)
        if self.cri_pix:
            l_pix = self.cri_pix(preds_student, self.gt)
            l_total += l_pix
            loss_dict['l_pix'] = l_pix
            
        # [Loss 2]: 學生與 GT 之間的 FFT Loss (全域頻域)
        if getattr(self, 'cri_fft', None):
            l_fft = self.cri_fft(preds_student, self.gt)
            l_total += l_fft
            loss_dict['l_fft'] = l_fft
            
        # [Loss 3]: 傳統 KD Loss (模仿老師輸出的 L1)
        l_kd = F.l1_loss(preds_student, preds_teacher) * self.kd_weight
        l_total += l_kd
        loss_dict['l_kd'] = l_kd

        # 📌 [Loss 4]: 小波引導蒸餾 Loss (Wavelet KD)
        # 讓學生針對水平、垂直、對角邊緣進行強化模仿
        l_wav_kd = self.wavelet_kd_loss(preds_student, preds_teacher) * self.kd_weight
        l_total += l_wav_kd
        loss_dict['l_wav_kd'] = l_wav_kd

        # 反向傳播與更新梯度
        l_total.backward()
        self.optimizer_g.step()
        
        self.log_dict = self.reduce_loss_dict(loss_dict)