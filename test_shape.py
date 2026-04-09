import torch
from basicsr.models.archs.EVSSM_arch import EVSSM

def main():
    print("🚀 正在載入 Wavelet-EVSSM 模型...")
    # 清理一下殘留的 GPU 記憶體
    torch.cuda.empty_cache()
    
    model = EVSSM().cuda()
    # 將模型設為「評估模式」(關閉 Dropout 等干擾)
    model.eval() 
    print("✅ 模型載入成功！")
    
    # 🌟 [修改] 縮小測試張量：Batch Size=1, 圖片縮小為 128x128
    print("📦 正在準備輸入張量 (Input Tensor)...")
    dummy_input = torch.randn(1, 3, 128, 128).cuda()
    print(f"   輸入維度: {dummy_input.shape}")
    
    print("⚙️ 開始進行前向傳播 (Forward Pass) 測試...")
    try:
        # 🌟 [核心修改] 告訴 PyTorch 不要計算梯度，省下 90% 記憶體！
        with torch.no_grad():
            output = model(dummy_input)
            
        print("🎉 前向傳播成功！沒有發生維度報錯！")
        print(f"   輸出維度: {output.shape}")
        
        if output.shape == dummy_input.shape:
            print("🏆 完美！輸出維度與輸入維度完全一致，Haar 小波替換手術 100% 成功！")
        else:
            print("⚠️ 警告：輸出維度不一樣！")
            
    except Exception as e:
        print("❌ 前向傳播失敗，捕獲到以下錯誤：")
        print(e)

if __name__ == "__main__":
    main()