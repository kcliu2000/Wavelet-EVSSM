import torch
from thop import profile
from thop import clever_format

# 引入你的 EVSSM 架構
from basicsr.models.archs.EVSSM_arch import EVSSM

def main():
    print("正在將模型載入至 GPU...")
    
    # 建立 48 Blocks 的老師模型 (加上 .cuda() 移至顯卡)
    teacher_model = EVSSM(
        inp_channels=3, out_channels=3, dim=48, 
        num_blocks=[6, 6, 12], ffn_expansion_factor=3, bias=False
    ).cuda()
    
    # 建立 24 Blocks 的學生模型 (加上 .cuda() 移至顯卡)
    student_model = EVSSM(
        inp_channels=3, out_channels=3, dim=48, 
        num_blocks=[3, 3, 6], ffn_expansion_factor=3, bias=False
    ).cuda()

    # 產生一張 Dummy 的測試圖片 (同樣加上 .cuda())
    dummy_input = torch.randn(1, 3, 256, 256).cuda()

    print("=========================================")
    print("🧠 正在計算 [原版 Teacher 模型] 的複雜度...")
    macs_t, params_t = profile(teacher_model, inputs=(dummy_input, ), verbose=False)
    macs_t_f, params_t_f = clever_format([macs_t, params_t], "%.3f")
    print(f"Teacher -> FLOPs (MACs): {macs_t_f}, 參數 (Params): {params_t_f}")

    print("-----------------------------------------")
    print("👶 正在計算 [剪枝 Student 模型] 的複雜度...")
    macs_s, params_s = profile(student_model, inputs=(dummy_input, ), verbose=False)
    macs_s_f, params_s_f = clever_format([macs_s, params_s], "%.3f")
    print(f"Student -> FLOPs (MACs): {macs_s_f}, 參數 (Params): {params_s_f}")
    
    print("=========================================")
    # 計算節省比例
    flops_saved = (1 - (macs_s / macs_t)) * 100
    params_saved = (1 - (params_s / params_t)) * 100
    print(f"🎉 你的貢獻：成功減少了 {flops_saved:.1f}% 的運算量 與 {params_saved:.1f}% 的參數量！")

if __name__ == '__main__':
    main()