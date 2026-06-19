import torch
import numpy as np

def print_wavelet_filters(pth_path):
    print(f"載入權重檔: {pth_path}\n")
    checkpoint = torch.load(pth_path, map_location='cpu')
    
    # BasicSR 通常把網路權重存在 params 或 params_ema 裡
    if 'params_ema' in checkpoint:
        state_dict = checkpoint['params_ema']
    elif 'params' in checkpoint:
        state_dict = checkpoint['params']
    else:
        state_dict = checkpoint
        
    # 抓出包含 filter 或 wave 關鍵字的權重
    filter_keys = [k for k in state_dict.keys() if 'filter' in k.lower() or 'wave' in k.lower()]
    
    if not filter_keys:
        print("❌ 找不到小波濾波器的權重，請確認檔名或模型架構。")
        return

    # 設定 numpy 的列印格式：保留 6 位小數，取消科學記號，方便閱讀
    np.set_printoptions(precision=6, suppress=True, linewidth=100)
    
    for key in filter_keys:
        weight = state_dict[key].squeeze().numpy()
        print(f"========== {key} ==========")
        print(weight)
        print()

if __name__ == "__main__":
    # ⚠️ 記得改成你實際的 .pth 檔案路徑！
    # 例如: 'experiments/EVSSM_Wavelet_GoPro/models/net_g_40000.pth'
    MODEL_PATH = '/home/m11302124/EVSSM/experiments/EVSSM_Wavelet_GoPro/models/net_g_120000.pth' 
    print_wavelet_filters(MODEL_PATH)