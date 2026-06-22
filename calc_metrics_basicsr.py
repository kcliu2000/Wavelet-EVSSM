import os
import cv2
import numpy as np
from tqdm import tqdm
from basicsr.metrics import calculate_psnr, calculate_ssim

pred_dir = "/home/m11302124/Thesis/EVSSM/results_final_2/output_kd/GoPro"
gt_dir = "/home/m11302124/MIMO-UNet-Wavelet/dataset/GOPRO/test/sharp"

psnr_list = []
ssim_list = []

file_list = sorted([
    f for f in os.listdir(pred_dir)
    if f.lower().endswith((".png", ".jpg", ".jpeg"))
])

for filename in tqdm(file_list):
    pred_path = os.path.join(pred_dir, filename)
    gt_path = os.path.join(gt_dir, filename)

    if not os.path.exists(gt_path):
        print(f"[Warning] GT not found: {gt_path}")
        continue

    pred = cv2.imread(pred_path, cv2.IMREAD_UNCHANGED)
    gt = cv2.imread(gt_path, cv2.IMREAD_UNCHANGED)

    if pred is None or gt is None:
        print(f"[Warning] Failed to read image: {filename}")
        continue

    if pred.shape != gt.shape:
        print(f"[Warning] Shape mismatch: {filename}, pred={pred.shape}, gt={gt.shape}")
        pred = cv2.resize(pred, (gt.shape[1], gt.shape[0]), interpolation=cv2.INTER_CUBIC)

    psnr_value = calculate_psnr(
        pred,
        gt,
        crop_border=0,
        input_order="HWC",
        test_y_channel=False
    )

    ssim_value = calculate_ssim(
        pred,
        gt,
        crop_border=0,
        input_order="HWC",
        test_y_channel=False
    )

    psnr_list.append(psnr_value)
    ssim_list.append(ssim_value)

print("======================================")
print(f"Number of evaluated images: {len(psnr_list)}")
print(f"Average PSNR: {np.mean(psnr_list):.4f} dB")
print(f"Average SSIM: {np.mean(ssim_list):.4f}")
print("======================================")