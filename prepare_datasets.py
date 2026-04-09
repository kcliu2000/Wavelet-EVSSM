import os
import cv2
import sys
import argparse
import numpy as np
from multiprocessing import Pool
from os import path as osp
from tqdm import tqdm

# 引入 BasicSR 核心工具 (請確保你在 EVSSM 根目錄執行)
from basicsr.utils import scandir
from basicsr.utils.lmdb_util import make_lmdb_from_imgs

# ============================================================
# 🛠️ 請修改這裡的路徑 🛠️
# ============================================================
# 指向你存放原始資料集 PNG 圖片的根目錄
DATASET_ROOT = '/home/m11302124/MLWNet-Baseline/datasets' 

# GoPro 原始路徑與輸出路徑
GOPRO_RAW = osp.join(DATASET_ROOT, 'GOPRO/train')
GOPRO_SUB = osp.join(DATASET_ROOT, 'GOPRO/train_sub')
GOPRO_LMDB = osp.join(DATASET_ROOT, 'GOPRO/train_lmdb')

# RealBlur 原始路徑與輸出路徑
REALBLUR_RAW = osp.join(DATASET_ROOT, 'RealBlur_J/train')
REALBLUR_LMDB = osp.join(DATASET_ROOT, 'RealBlur_J/train_lmdb')
# ============================================================

def worker(path, opt):
    """子圖切割 worker (改編自 extract_subimages.py)"""
    crop_size = opt['crop_size']
    step = opt['step']
    thresh_size = opt['thresh_size']
    img_name, extension = osp.splitext(osp.basename(path))
    
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    h, w = img.shape[0:2]
    h_space = np.arange(0, h - crop_size + 1, step)
    if h - (h_space[-1] + crop_size) > thresh_size:
        h_space = np.append(h_space, h - crop_size)
    w_space = np.arange(0, w - crop_size + 1, step)
    if w - (w_space[-1] + crop_size) > thresh_size:
        w_space = np.append(w_space, w - crop_size)

    index = 0
    for x in h_space:
        for y in w_space:
            index += 1
            cropped_img = img[x:x + crop_size, y:y + crop_size, ...]
            cropped_img = np.ascontiguousarray(cropped_img)
            cv2.imwrite(
                osp.join(opt['save_folder'], f'{img_name}_s{index:03d}{extension}'), 
                cropped_img, [cv2.IMWRITE_PNG_COMPRESSION, opt['compression_level']])

def run_extract_subimages(input_dir, save_dir, crop_size=480, step=240):
    
    """執行切圖流程"""
    if not osp.exists(save_dir):
        os.makedirs(save_dir)
    
    opt = {
        'input_folder': input_dir,
        'save_folder': save_dir,
        'crop_size': crop_size,
        'step': step,
        'thresh_size': 0,
        'compression_level': 3,
        'n_thread': 20
    }
    
    img_list = list(scandir(input_dir, full_path=True))
    pbar = tqdm(total=len(img_list), desc=f'Extracting {osp.basename(input_dir)}')
    pool = Pool(opt['n_thread'])
    for path in img_list:
        pool.apply_async(worker, args=(path, opt), callback=lambda arg: pbar.update(1))
    pool.close()
    pool.join()
    pbar.close()

def run_create_lmdb(folder_path, lmdb_path):
    os.makedirs(osp.dirname(lmdb_path), exist_ok=True)
    """執行 LMDB 轉換"""
    print(f'Reading image path list from {folder_path} ...')
    img_path_list = sorted(list(scandir(folder_path, suffix='png', recursive=False)))
    keys = [img_path.split('.png')[0] for img_path in img_path_list]
    make_lmdb_from_imgs(folder_path, lmdb_path, img_path_list, keys, multiprocessing_read=True)

def process_gopro():
    """GoPro: 先切圖再轉 LMDB"""
    print("=== Processing GoPro Dataset ===")
    for mode in ['sharp', 'blur']:
        raw_dir = osp.join(GOPRO_RAW, mode)
        sub_dir = osp.join(GOPRO_SUB, mode)
        lmdb_dir = osp.join(GOPRO_LMDB, f'{mode}_crops.lmdb')
        
        print(f"1. Extracting {mode} sub-images...")
        run_extract_subimages(raw_dir, sub_dir)
        
        print(f"2. Creating {mode} LMDB...")
        run_create_lmdb(sub_dir, lmdb_dir)

def process_realblur():
    """RealBlur: 直接轉 LMDB"""
    print("=== Processing RealBlur-J Dataset ===")
    for mode in ['sharp', 'blur']:
        raw_dir = osp.join(REALBLUR_RAW, mode)
        lmdb_dir = osp.join(REALBLUR_LMDB, f'{mode}.lmdb')
        
        print(f"Creating {mode} LMDB directly...")
        run_create_lmdb(raw_dir, lmdb_dir)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, choices=['gopro', 'realblur', 'all'], required=True)
    args = parser.parse_args()

    if args.dataset in ['gopro', 'all']:
        process_gopro()
    if args.dataset in ['realblur', 'all']:
        process_realblur()
    
    print("\n✅ 所有資料準備工作已完成！")