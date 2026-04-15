import torch
import torch.nn.functional as F
from collections import OrderedDict
from basicsr.models.image_restoration_model import ImageRestorationModel
from basicsr.models.archs.EVSSM_arch import EVSSM

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

        # 📌 終極修正：手動載入 FFTLoss 避開 build_loss 錯誤
        self.cri_fft = None
        if opt['train'].get('fft_opt'):
            # 檢查父類別是否已經偷偷幫我們建好了
            if hasattr(self, 'cri_fft') and self.cri_fft is not None:
                pass 
            else:
                try:
                    # 直接從檔案引入，繞過註冊表
                    from basicsr.models.losses.losses import FFTLoss
                    fft_opt = opt['train']['fft_opt'].copy()
                    fft_opt.pop('type', None) # 把 'type' 刪掉，只留參數
                    self.cri_fft = FFTLoss(**fft_opt).to(self.device)
                except ImportError:
                    print("警告: 找不到 FFTLoss，將僅使用 L1 + KD Loss。")

    def optimize_parameters(self, current_iter, tb_logger=None):
        self.optimizer_g.zero_grad()
        
        preds_student = self.net_g(self.lq)
        
        with torch.no_grad():
            preds_teacher = self.net_teacher(self.lq)
            
        l_total = 0
        loss_dict = OrderedDict()
        
        # 1. Pixel Loss (L1)
        if self.cri_pix:
            l_pix = self.cri_pix(preds_student, self.gt)
            l_total += l_pix
            loss_dict['l_pix'] = l_pix
            
        # 📌 2. FFT Loss (頻域)
        if getattr(self, 'cri_fft', None):
            l_fft = self.cri_fft(preds_student, self.gt)
            l_total += l_fft
            loss_dict['l_fft'] = l_fft
            
        # 3. Knowledge Distillation Loss (模仿老師)
        l_kd = F.l1_loss(preds_student, preds_teacher) * self.kd_weight
        l_total += l_kd
        loss_dict['l_kd'] = l_kd

        l_total.backward()
        self.optimizer_g.step()
        
        self.log_dict = self.reduce_loss_dict(loss_dict)