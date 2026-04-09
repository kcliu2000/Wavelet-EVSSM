import os
import cv2
import sys
import argparse
import numpy as np
from multiprocessing import Pool
from os import path as osp
from tqdm import tqdm

# 引入 BasicSR 核心工具
from basicsr.utils import scandir
from basicsr.utils.lmdb_util import make_lmdb_from_imgs

# ============================================================
# 🛠️ 路徑設定 (請根據你的 ib701 環境修改)
# ============================================================
# GoPro
GOPRO_RAW_DIR = '/home/m11302124/MIMO-UNet-Wavelet/dataset/GOPRO'
GOPRO_OUT_DIR = '/home/m11302124/MIMO-UNet-Wavelet/dataset/GOPRO'

# RealBlur-J
REALBLUR_RAW_DIR = '/home/m11302124/MLWNet-Baseline/datasets/RealBlur_J'
REALBLUR_OUT_DIR = '/home/m11302124/MLWNet-Baseline/datasets/RealBlur_J'
# ============================================================

def is_lmdb_finished(lmdb_path):
    """檢查 LMDB 是否已經製作完成且包含數據"""
    # LMDB 通常是一個資料夾，裡面有 data.mdb 檔案
    data_file = osp.join(lmdb_path, 'data.mdb')
    return osp.exists(data_file) and osp.getsize(data_file) > 1024 * 1024 # 大於 1MB 視為有效

def run_extract_subimages(input_dir, save_dir, crop_size=480, step=240):
    """執行切圖，若已有子圖則跳過"""
    if osp.exists(save_dir) and len(os.listdir(save_dir)) > 100:
        print(f"⏩ 子圖已存在於 {save_dir}，跳過切割階段。")
        return True

    os.makedirs(save_dir, exist_ok=True)
    img_list = list(scandir(input_dir, full_path=True))
    if not img_list:
        print(f"⚠️ 找不到原始圖片：{input_dir}")
        return False

    opt = {'input_folder': input_dir, 'save_folder': save_dir, 'crop_size': crop_size, 
           'step': step, 'thresh_size': 0, 'compression_level': 3, 'n_thread': 20}
    
    pbar = tqdm(total=len(img_list), desc=f'Extracting {osp.basename(input_dir)}')
    pool = Pool(opt['n_thread'])
    for path in img_list:
        pool.apply_async(worker, args=(path, opt), callback=lambda arg: pbar.update(1))
    pool.close(); pool.join(); pbar.close()
    return True

def worker(path, opt):
    """子圖切割核心邏輯"""
    crop_size, step, thresh_size = opt['crop_size'], opt['step'], opt['thresh_size']
    img_name, extension = osp.splitext(osp.basename(path))
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    h, w = img.shape[0:2]
    h_space = np.append(np.arange(0, h - crop_size + 1, step), h - crop_size)
    w_space = np.append(np.arange(0, w - crop_size + 1, step), w - crop_size)
    index = 0
    for x in h_space:
        for y in w_space:
            index += 1
            cropped_img = np.ascontiguousarray(img[x:x + crop_size, y:y + crop_size, ...])
            cv2.imwrite(osp.join(opt['save_folder'], f'{img_name}_s{index:03d}{extension}'), 
                        cropped_img, [cv2.IMWRITE_PNG_COMPRESSION, opt['compression_level']])

def run_create_lmdb(folder_path, lmdb_path):
    """建立 LMDB"""
    if is_lmdb_finished(lmdb_path):
        print(f"✅ LMDB 已存在：{lmdb_path}，跳過封裝。")
        return

    os.makedirs(osp.dirname(lmdb_path), exist_ok=True)
    print(f'🚀 正在建立 LMDB：{lmdb_path}')
    img_path_list = sorted(list(scandir(folder_path, suffix='png', recursive=False)))
    keys = [img_path.split('.png')[0] for img_path in img_path_list]
    make_lmdb_from_imgs(folder_path, lmdb_path, img_path_list, keys, multiprocessing_read=True)

def process_gopro():
    """GoPro 專屬流程"""
    print("\n=== ⚙️ Checking GoPro Dataset ===")
    for phase in ['train', 'test']:
        for mode in ['sharp', 'blur']:
            lmdb_path = osp.join(GOPRO_OUT_DIR, f'{phase}_lmdb', f'{mode}_crops.lmdb')
            
            if is_lmdb_finished(lmdb_path):
                print(f"✨ GoPro {phase} {mode} 已經全部做好了，跳過。")
                continue
            
            raw_dir = osp.join(GOPRO_RAW_DIR, phase, mode)
            sub_dir = osp.join(GOPRO_OUT_DIR, f'{phase}_sub', mode)
            
            # 只有在 LMDB 不存在時才檢查並執行切圖與封裝
            if run_extract_subimages(raw_dir, sub_dir):
                run_create_lmdb(sub_dir, lmdb_path)

def process_realblur():
    """RealBlur 專屬流程"""
    print("\n=== ⚙️ Checking RealBlur-J Dataset ===")
    for phase in ['train', 'test']:
        for mode in ['sharp', 'blur']:
            lmdb_path = osp.join(REALBLUR_OUT_DIR, f'{phase}_lmdb', f'{mode}.lmdb')
            
            if is_lmdb_finished(lmdb_path):
                print(f"✨ RealBlur {phase} {mode} 已經做好了，跳過。")
                continue
            
            raw_dir = osp.join(REALBLUR_RAW_DIR, phase, mode)
            run_create_lmdb(raw_dir, lmdb_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, choices=['gopro', 'realblur', 'all'], required=True)
    args = parser.parse_args()
    
    if args.dataset in ['gopro', 'all']: process_gopro()
    if args.dataset in ['realblur', 'all']: process_realblur()
    print("\n✅ 智慧檢查與補齊工作完成！")