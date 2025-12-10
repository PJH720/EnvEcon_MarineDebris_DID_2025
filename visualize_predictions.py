# -*- coding: utf-8 -*-
'''
발표용 예측 결과 시각화 스크립트
원본 이미지, 정답 마스크, 예측 마스크를 나란히 비교하여 시각화합니다.
'''

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap
import rasterio
from tqdm import tqdm
from os.path import dirname as up

# 프로젝트 루트 경로 추가
script_dir = os.path.dirname(os.path.abspath(__file__))
root_path = script_dir  # 스크립트가 프로젝트 루트에 있음
if root_path not in sys.path:
    sys.path.insert(0, root_path)
utils_path = os.path.join(root_path, 'utils')
if utils_path not in sys.path:
    sys.path.insert(0, utils_path)

# 직접 import
import importlib.util
assets_file = os.path.join(utils_path, "assets.py")
spec = importlib.util.spec_from_file_location("assets", assets_file)
assets = importlib.util.module_from_spec(spec)
spec.loader.exec_module(assets)
labels = assets.labels
color_mapping = assets.color_mapping
cat_mapping = assets.cat_mapping
s2_mapping = assets.s2_mapping

def color_name_to_rgb(color_name):
    """색상 이름을 RGB 값으로 변환"""
    color_map = {
        'red': [1, 0, 0],
        'green': [0, 1, 0],
        'limegreen': [0.2, 1, 0.2],
        'navy': [0, 0, 0.5],
        'purple': [0.5, 0, 0.5],
        'silver': [0.75, 0.75, 0.75],
        'gray': [0.5, 0.5, 0.5],
        'brown': [0.6, 0.4, 0.2],
        'orange': [1, 0.65, 0],
        'yellow': [1, 1, 0],
        'darkturquoise': [0, 0.8, 0.82],
        'darkkhaki': [0.74, 0.72, 0.42],
        'gold': [1, 0.84, 0],
        'seashell': [1, 0.96, 0.93],
        'rosybrown': [0.74, 0.56, 0.56]
    }
    return color_map.get(color_name, [1, 1, 1])

def create_rgb_image(image):
    """Sentinel-2 이미지에서 RGB 이미지 생성 (밴드 7, 3, 2 = nm842, nm560, nm490)"""
    # 이미지가 (C, H, W) 형태인지 (H, W, C) 형태인지 확인
    if len(image.shape) == 3:
        if image.shape[0] == 11:  # (C, H, W)
            rgb = np.stack([
                image[s2_mapping['nm842']],  # R
                image[s2_mapping['nm560']],  # G
                image[s2_mapping['nm490']]   # B
            ], axis=-1)
        else:  # (H, W, C)
            rgb = np.stack([
                image[:, :, s2_mapping['nm842']],
                image[:, :, s2_mapping['nm560']],
                image[:, :, s2_mapping['nm490']]
            ], axis=-1)
    else:
        raise ValueError(f"Unexpected image shape: {image.shape}")
    
    # 정규화 (0-1 범위)
    rgb = np.clip(rgb, 0, 1)
    
    # 스트레칭 (2% 선형 스트레칭)
    for i in range(3):
        band = rgb[:, :, i]
        valid_pixels = band[band > 0]
        if len(valid_pixels) > 0:
            p2, p98 = np.percentile(valid_pixels, [2, 98])
            if p98 > p2:
                rgb[:, :, i] = np.clip((band - p2) / (p98 - p2 + 1e-10), 0, 1)
    
    return rgb

def create_colored_mask(mask, agg_to_water=True):
    """마스크를 색상으로 변환"""
    # agg_to_water가 True인 경우 클래스 12, 13, 14, 15를 7로 변환
    if agg_to_water:
        mask_agg = mask.copy()
        mask_agg[mask_agg == 15] = 7  # Mixed Water
        mask_agg[mask_agg == 14] = 7  # Wakes
        mask_agg[mask_agg == 13] = 7  # Cloud Shadows
        mask_agg[mask_agg == 12] = 7  # Waves
        mask = mask_agg
    
    # 클래스 인덱스를 0부터 시작하도록 변환 (1-15 -> 0-14)
    mask_zero = mask - 1
    mask_zero[mask_zero < 0] = -1  # 배경은 -1로 유지
    
    # 색상 맵 생성
    unique_classes = np.unique(mask[mask > 0])
    colored_mask = np.zeros((mask.shape[0], mask.shape[1], 3))
    
    # agg_to_water가 True인 경우 labels도 조정
    if agg_to_water:
        labels_used = labels[:-4]  # Mixed Water, Wakes, Cloud Shadows, Waves 제거
    else:
        labels_used = labels
    
    for cls_val in unique_classes:
        if cls_val <= len(labels_used):
            class_name = labels_used[cls_val - 1]
            color = color_mapping.get(class_name, 'white')
            rgb_color = color_name_to_rgb(color)
            colored_mask[mask == cls_val] = rgb_color
    
    return colored_mask, labels_used

def visualize_prediction(original_path, gt_mask_path, pred_mask_path, output_path, agg_to_water=True):
    """단일 예측 결과 시각화"""
    # 원본 이미지 로드
    with rasterio.open(original_path) as src:
        original = src.read()  # (C, H, W)
    
    # 정답 마스크 로드
    with rasterio.open(gt_mask_path) as src:
        gt_mask = src.read(1)  # (H, W)
    
    # 예측 마스크 로드
    with rasterio.open(pred_mask_path) as src:
        pred_mask = src.read(1)  # (H, W)
    
    # RGB 이미지 생성
    rgb = create_rgb_image(original)
    
    # 색상 마스크 생성
    gt_colored, labels_used = create_colored_mask(gt_mask, agg_to_water)
    pred_colored, _ = create_colored_mask(pred_mask, agg_to_water)
    
    # 시각화
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # 원본 RGB 이미지
    axes[0].imshow(rgb)
    axes[0].set_title('Original RGB Image', fontsize=14, fontweight='bold')
    axes[0].axis('off')
    
    # 정답 마스크
    axes[1].imshow(rgb)
    axes[1].imshow(gt_colored, alpha=0.6)
    axes[1].set_title('Ground Truth Mask', fontsize=14, fontweight='bold')
    axes[1].axis('off')
    
    # 예측 마스크
    axes[2].imshow(rgb)
    axes[2].imshow(pred_colored, alpha=0.6)
    axes[2].set_title('Predicted Mask (U-Net)', fontsize=14, fontweight='bold')
    axes[2].axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

def main():
    """메인 함수"""
    # 경로 설정
    data_path = os.path.join(root_path, 'data')
    patches_path = os.path.join(data_path, 'patches')
    predicted_path = os.path.join(data_path, 'predicted_unet')
    output_path = os.path.join(data_path, 'presentation', 'visualizations')
    
    # 출력 디렉토리 생성
    os.makedirs(output_path, exist_ok=True)
    
    # 예측 결과가 있는지 확인
    if not os.path.exists(predicted_path):
        print(f"경고: 예측 결과 폴더가 없습니다: {predicted_path}")
        print("먼저 evaluation.py를 실행하여 예측 마스크를 생성하세요.")
        return
    
    # 테스트 세트 파일 목록 로드
    test_file = os.path.join(data_path, 'splits', 'test_X.txt')
    if not os.path.exists(test_file):
        print(f"경고: 테스트 세트 파일이 없습니다: {test_file}")
        return
    
    test_rois = np.genfromtxt(test_file, dtype='str')
    
    # 각 예측 결과 시각화
    print("예측 결과 시각화 중...")
    for roi in tqdm(test_rois):
        # 파일 경로 구성
        roi_folder = '_'.join(['S2'] + roi.split('_')[:-1])
        roi_name = '_'.join(['S2'] + roi.split('_'))
        
        original_file = os.path.join(patches_path, roi_folder, roi_name + '.tif')
        gt_mask_file = os.path.join(patches_path, roi_folder, roi_name + '_cl.tif')
        pred_mask_file = os.path.join(predicted_path, roi_name + '_unet.tif')
        
        # 파일 존재 확인
        if not all(os.path.exists(f) for f in [original_file, gt_mask_file, pred_mask_file]):
            continue
        
        # 출력 파일 경로
        output_file = os.path.join(output_path, roi_name + '_visualization.png')
        
        # 시각화
        try:
            visualize_prediction(original_file, gt_mask_file, pred_mask_file, output_file)
        except Exception as e:
            print(f"오류 발생 ({roi_name}): {e}")
            continue
    
    print(f"\n시각화 완료! 결과는 {output_path}에 저장되었습니다.")

if __name__ == "__main__":
    main()

