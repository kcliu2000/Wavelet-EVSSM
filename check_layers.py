# check_layers.py
from basicsr.models.archs.EVSSM_arch import EVSSM

# 建立一個測試用的學生模型
model = EVSSM(inp_channels=3, out_channels=3, dim=48, num_blocks=[3, 3, 6])

print("=== 網路層清單 ===")
for name, module in model.named_modules():
    # 只印出帶有 Block 或 Stage 關鍵字的層，避免印出太多細節
    if 'block' in name.lower() or 'down' in name.lower() or 'up' in name.lower():
        print(name)