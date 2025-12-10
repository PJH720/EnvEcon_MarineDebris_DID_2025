# -*- coding: utf-8 -*-
'''
MARIDA 데이터셋 탐색적 데이터 분석 (EDA) 리포트 생성 스크립트
'''

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from glob import glob
from osgeo import gdal
from os.path import dirname as up

# 프로젝트 루트 경로
root_path = Path(__file__).parent
sys.path.append(str(root_path / 'utils'))
from assets import (cat_mapping, labels, color_mapping, s2_mapping, 
                   conf_mapping, roi_mapping)

# 한글 폰트 설정 (Windows)
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

# 출력 디렉토리
output_dir = root_path / 'data' / 'eda_images'
output_dir.mkdir(exist_ok=True, parents=True)

# 역 매핑 생성
rev_cat_mapping = {v: k for k, v in cat_mapping.items()}
rev_conf_mapping = {v: k for k, v in conf_mapping.items()}

def load_patch(patch_path):
    """패치 파일 로드 (이미지, 클래스 마스크, 신뢰도 마스크)"""
    patch_path = Path(patch_path)
    
    # 패치 이미지 로드
    ds = gdal.Open(str(patch_path))
    if ds is None:
        raise ValueError(f"Cannot open patch: {patch_path}")
    
    image = np.copy(ds.ReadAsArray())  # Shape: (bands, height, width)
    ds = None
    
    # 파일명에서 기본 이름 추출
    base_name = patch_path.stem
    parts = base_name.split('_')
    base_prefix = '_'.join(parts[:4])  # S2_DATE_TILE_CROP
    
    patch_dir = patch_path.parent
    
    # 클래스 마스크 로드
    cl_path = patch_dir / f"{base_prefix}_cl.tif"
    if cl_path.exists():
        ds_cl = gdal.Open(str(cl_path))
        class_mask = np.copy(ds_cl.ReadAsArray())  # Shape: (height, width)
        ds_cl = None
    else:
        class_mask = None
    
    # 신뢰도 마스크 로드
    conf_path = patch_dir / f"{base_prefix}_conf.tif"
    if conf_path.exists():
        ds_conf = gdal.Open(str(conf_path))
        conf_mask = np.copy(ds_conf.ReadAsArray())  # Shape: (height, width)
        ds_conf = None
    else:
        conf_mask = None
    
    return image, class_mask, conf_mask

def create_rgb_image(image, rgb_bands=None):
    """RGB 이미지 생성"""
    if rgb_bands is None:
        # 표준 RGB: nm665(R), nm560(G), nm490(B)
        rgb_bands = ['nm665', 'nm560', 'nm490']
    
    band_indices = [s2_mapping[band] for band in rgb_bands]
    rgb = np.stack([image[i] for i in band_indices], axis=-1)
    
    # 정규화 (0-1 범위)
    rgb = np.clip(rgb, 0, 1)
    
    # 스트레칭 (2% 선형 스트레칭)
    for i in range(3):
        band = rgb[:, :, i]
        p2, p98 = np.percentile(band[band > 0], [2, 98])
        rgb[:, :, i] = np.clip((band - p2) / (p98 - p2 + 1e-10), 0, 1)
    
    return rgb

def create_false_color_image(image):
    """False Color 이미지 생성 (NIR, Red Edge, Green)"""
    # False Color: nm842(NIR), nm740(Red Edge), nm560(Green)
    false_color_bands = ['nm842', 'nm740', 'nm560']
    return create_rgb_image(image, false_color_bands)

def visualize_patch(image, class_mask=None, conf_mask=None, title="", save_path=None):
    """패치 시각화 (RGB, 클래스 마스크 오버레이)"""
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # RGB 이미지
    rgb = create_rgb_image(image)
    axes[0].imshow(rgb)
    axes[0].set_title(f'{title}\nRGB (nm665, nm560, nm490)', fontsize=12)
    axes[0].axis('off')
    
    # False Color 이미지
    false_color = create_false_color_image(image)
    axes[1].imshow(false_color)
    axes[1].set_title('False Color (nm842, nm740, nm560)', fontsize=12)
    axes[1].axis('off')
    
    # 클래스 마스크 오버레이
    axes[2].imshow(rgb)
    if class_mask is not None:
        # 클래스 마스크를 컬러로 오버레이
        mask_colored = np.zeros((*class_mask.shape, 3))
        unique_classes = np.unique(class_mask)
        unique_classes = unique_classes[unique_classes > 0]  # 0 제외
        
        for cls_val in unique_classes:
            if cls_val in rev_cat_mapping:
                class_name = rev_cat_mapping[cls_val]
                color = color_mapping.get(class_name, 'white')
                # 컬러 이름을 RGB로 변환
                if color == 'red':
                    rgb_color = [1, 0, 0]
                elif color == 'green':
                    rgb_color = [0, 1, 0]
                elif color == 'limegreen':
                    rgb_color = [0.2, 1, 0.2]
                elif color == 'navy':
                    rgb_color = [0, 0, 0.5]
                elif color == 'purple':
                    rgb_color = [0.5, 0, 0.5]
                elif color == 'silver':
                    rgb_color = [0.75, 0.75, 0.75]
                elif color == 'gray':
                    rgb_color = [0.5, 0.5, 0.5]
                elif color == 'brown':
                    rgb_color = [0.6, 0.4, 0.2]
                elif color == 'orange':
                    rgb_color = [1, 0.65, 0]
                elif color == 'yellow':
                    rgb_color = [1, 1, 0]
                elif color == 'darkturquoise':
                    rgb_color = [0, 0.8, 0.82]
                elif color == 'darkkhaki':
                    rgb_color = [0.74, 0.72, 0.42]
                elif color == 'gold':
                    rgb_color = [1, 0.84, 0]
                elif color == 'seashell':
                    rgb_color = [1, 0.96, 0.93]
                elif color == 'rosybrown':
                    rgb_color = [0.74, 0.56, 0.56]
                else:
                    rgb_color = [1, 1, 1]
                
                mask_colored[class_mask == cls_val] = rgb_color
        
        axes[2].imshow(mask_colored, alpha=0.5)
        axes[2].set_title('RGB with Class Mask Overlay', fontsize=12)
    else:
        axes[2].set_title('RGB (no mask)', fontsize=12)
    axes[2].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()
    
    return fig

def select_representative_patches(df_train, df_val, df_test, classes=None, patches_per_class=2):
    """클래스별 대표 패치 선택"""
    df_all = pd.concat([df_train, df_val, df_test], ignore_index=True)
    
    if classes is None:
        # 주요 클래스 선택
        class_counts = df_all['Class'].value_counts()
        classes = class_counts.head(10).index.tolist()  # 상위 10개 클래스
    
    selected_patches = {}
    
    for class_name in classes:
        class_data = df_all[df_all['Class'] == class_name]
        
        if len(class_data) == 0:
            continue
        
        # 패치별 픽셀 수 계산
        patch_counts = class_data.groupby(['Date', 'Tile', 'Image']).size().reset_index(name='count')
        patch_counts = patch_counts.sort_values('count', ascending=False)
        
        # 상위 패치 선택
        selected = patch_counts.head(patches_per_class)
        
        patches = []
        for _, row in selected.iterrows():
            patch_id = f"{row['Date']}_{row['Tile']}_{row['Image']}"
            patches.append({
                'patch_id': patch_id,
                'date': row['Date'],
                'tile': row['Tile'],
                'image': row['Image'],
                'pixel_count': row['count']
            })
        
        if patches:
            selected_patches[class_name] = patches
    
    return selected_patches

def get_patch_path(patch_info):
    """패치 정보로부터 파일 경로 생성"""
    date = patch_info['date']
    tile = patch_info['tile']
    image = patch_info['image']
    
    folder_name = f"S2_{date}_{tile}"
    file_name = f"S2_{date}_{tile}_{image}.tif"
    
    patch_path = root_path / 'data' / 'patches' / folder_name / file_name
    
    return patch_path if patch_path.exists() else None

def create_histograms(df_train, df_val, df_test, save_dir):
    """스펙트럴 밴드별 히스토그램 생성"""
    df_all = pd.concat([df_train, df_val, df_test], ignore_index=True)
    
    band_columns = sorted([col for col in df_all.columns if col.startswith('nm')])
    
    # 전체 데이터셋 히스토그램
    fig, axes = plt.subplots(3, 4, figsize=(16, 12))
    axes = axes.flatten()
    
    for idx, band in enumerate(band_columns[:11]):  # 11개 밴드
        ax = axes[idx]
        data = df_all[band].values
        data = data[data > 0]  # 0 제외
        
        ax.hist(data, bins=100, alpha=0.7, edgecolor='black')
        ax.set_title(f'{band}', fontsize=10)
        ax.set_xlabel('Reflectance')
        ax.set_ylabel('Frequency')
        ax.grid(True, alpha=0.3)
    
    # 마지막 subplot 제거
    axes[11].axis('off')
    
    plt.suptitle('Spectral Band Histograms (All Data)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_dir / 'histograms_all_bands.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # 주요 클래스별 히스토그램 (선택된 밴드)
    main_classes = ['Marine Debris', 'Dense Sargassum', 'Marine Water', 'Clouds']
    selected_bands = ['nm490', 'nm665', 'nm842', 'nm1600']
    
    fig, axes = plt.subplots(len(main_classes), len(selected_bands), 
                            figsize=(16, 12))
    
    for i, class_name in enumerate(main_classes):
        class_data = df_all[df_all['Class'] == class_name]
        
        if len(class_data) == 0:
            for j in range(len(selected_bands)):
                axes[i, j].axis('off')
            continue
        
        for j, band in enumerate(selected_bands):
            ax = axes[i, j]
            data = class_data[band].values
            data = data[data > 0]
            
            if len(data) > 0:
                ax.hist(data, bins=50, alpha=0.7, edgecolor='black',
                       color=color_mapping.get(class_name, 'gray'))
                ax.set_title(f'{class_name}\n{band}', fontsize=9)
                ax.set_xlabel('Reflectance')
                ax.set_ylabel('Frequency')
                ax.grid(True, alpha=0.3)
            else:
                ax.axis('off')
    
    plt.suptitle('Class-wise Spectral Band Histograms', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_dir / 'histograms_by_class.png', dpi=150, bbox_inches='tight')
    plt.close()

def create_spectral_signature_plot(df_train, df_val, df_test, save_dir):
    """클래스별 평균 스펙트럴 시그니처 플롯"""
    df_all = pd.concat([df_train, df_val, df_test], ignore_index=True)
    
    band_columns = sorted([col for col in df_all.columns if col.startswith('nm')])
    wavelengths = [int(col.replace('nm', '')) for col in band_columns]
    
    # 클래스별 평균 스펙트럴 시그니처
    class_spectra = df_all.groupby('Class')[band_columns].mean()
    
    # 주요 클래스 선택
    main_classes = ['Marine Debris', 'Dense Sargassum', 'Sparse Sargassum',
                   'Marine Water', 'Clouds', 'Ship', 'Turbid Water']
    
    plt.figure(figsize=(12, 8))
    
    for class_name in main_classes:
        if class_name in class_spectra.index:
            spectrum = class_spectra.loc[class_name, band_columns].values
            color = color_mapping.get(class_name, 'gray')
            plt.plot(wavelengths, spectrum, marker='o', label=class_name,
                    color=color, linewidth=2, markersize=6)
    
    plt.xlabel('Wavelength (nm)', fontsize=12)
    plt.ylabel('Mean Reflectance', fontsize=12)
    plt.title('Average Spectral Signatures by Class', fontsize=14, fontweight='bold')
    plt.legend(loc='best', fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_dir / 'spectral_signatures.png', dpi=150, bbox_inches='tight')
    plt.close()

def generate_markdown_report(df_train, df_val, df_test, selected_patches, 
                            patch_images, save_path):
    """마크다운 리포트 생성"""
    
    df_all = pd.concat([df_train, df_val, df_test], ignore_index=True)
    
    with open(save_path, 'w', encoding='utf-8') as f:
        f.write("# MARIDA 데이터셋 탐색적 데이터 분석 (EDA) 리포트\n\n")
        f.write("## 1. 데이터셋 개요\n\n")
        
        f.write(f"- **총 픽셀 수**: {len(df_all):,}\n")
        f.write(f"- **Train**: {len(df_train):,} ({len(df_train)/len(df_all)*100:.2f}%)\n")
        f.write(f"- **Val**: {len(df_val):,} ({len(df_val)/len(df_all)*100:.2f}%)\n")
        f.write(f"- **Test**: {len(df_test):,} ({len(df_test)/len(df_all)*100:.2f}%)\n\n")
        
        f.write("### 1.1 스펙트럴 밴드 정보\n\n")
        f.write("Sentinel-2 밴드:\n")
        band_columns = sorted([col for col in df_all.columns if col.startswith('nm')])
        for band in band_columns:
            wavelength = int(band.replace('nm', ''))
            f.write(f"- {band}: {wavelength}nm\n")
        f.write("\n")
        
        f.write("## 2. 클래스별 통계\n\n")
        
        class_counts = df_all['Class'].value_counts()
        f.write("### 2.1 클래스별 픽셀 분포\n\n")
        f.write("| 클래스 | 픽셀 수 | 비율 (%) |\n")
        f.write("|--------|---------|----------|\n")
        
        for class_name, count in class_counts.items():
            percentage = (count / len(df_all)) * 100
            f.write(f"| {class_name} | {count:,} | {percentage:.2f} |\n")
        f.write("\n")
        
        f.write("### 2.2 Split별 클래스 분포\n\n")
        f.write("| 클래스 | Train | Val | Test |\n")
        f.write("|--------|-------|-----|------|\n")
        
        for class_name in class_counts.index:
            train_count = len(df_train[df_train['Class'] == class_name])
            val_count = len(df_val[df_val['Class'] == class_name])
            test_count = len(df_test[df_test['Class'] == class_name])
            f.write(f"| {class_name} | {train_count:,} | {val_count:,} | {test_count:,} |\n")
        f.write("\n")
        
        f.write("## 3. 대표 패치 시각화\n\n")
        
        for class_name, patches in selected_patches.items():
            f.write(f"### 3.{list(selected_patches.keys()).index(class_name) + 1} {class_name}\n\n")
            
            for i, patch_info in enumerate(patches):
                patch_id = patch_info['patch_id']
                if patch_id in patch_images:
                    img_path = patch_images[patch_id]
                    rel_path = f"eda_images/{Path(img_path).name}"
                    f.write(f"#### 패치 {i+1}: {patch_id}\n\n")
                    f.write(f"- **픽셀 수**: {patch_info['pixel_count']:,}\n")
                    f.write(f"- **날짜**: {patch_info['date']}\n")
                    f.write(f"- **타일**: {patch_info['tile']}\n\n")
                    f.write(f"![{patch_id}]({rel_path})\n\n")
        f.write("\n")
        
        f.write("## 4. 스펙트럴 밴드 히스토그램\n\n")
        f.write("### 4.1 전체 데이터셋 밴드별 히스토그램\n\n")
        f.write("![All Bands Histogram](eda_images/histograms_all_bands.png)\n\n")
        
        f.write("### 4.2 클래스별 밴드별 히스토그램\n\n")
        f.write("![Class-wise Histogram](eda_images/histograms_by_class.png)\n\n")
        
        f.write("## 5. 클래스별 스펙트럴 시그니처\n\n")
        f.write("![Spectral Signatures](eda_images/spectral_signatures.png)\n\n")
        
        f.write("## 6. 특이 밴드 분석\n\n")
        
        # 각 클래스별로 특이 밴드 찾기
        class_spectra = df_all.groupby('Class')[band_columns].mean()
        
        f.write("### 6.1 클래스별 평균 반사율 (상위 3개 밴드)\n\n")
        f.write("| 클래스 | 1위 밴드 | 2위 밴드 | 3위 밴드 |\n")
        f.write("|--------|----------|----------|----------|\n")
        
        for class_name in class_counts.index[:10]:  # 상위 10개 클래스
            if class_name in class_spectra.index:
                spectrum = class_spectra.loc[class_name, band_columns]
                top3 = spectrum.nlargest(3)
                bands_str = " | ".join([f"{band} ({val:.4f})" for band, val in top3.items()])
                f.write(f"| {class_name} | {bands_str} |\n")
        f.write("\n")
        
        f.write("---\n\n")
        f.write("*이 리포트는 자동으로 생성되었습니다.*\n")

def main():
    """메인 함수"""
    print("=" * 60)
    print("MARIDA 데이터셋 EDA 리포트 생성")
    print("=" * 60)
    
    # 데이터 로드
    print("\n[1/6] 데이터셋 로드 중...")
    hdf_path = root_path / 'data' / 'dataset.h5'
    
    if not hdf_path.exists():
        raise FileNotFoundError(f"dataset.h5 파일을 찾을 수 없습니다: {hdf_path}")
    
    hdf = pd.HDFStore(str(hdf_path), mode='r')
    df_train = hdf.select('train')
    df_val = hdf.select('val')
    df_test = hdf.select('test')
    hdf.close()
    
    print(f"  - Train: {len(df_train):,} 픽셀")
    print(f"  - Val: {len(df_val):,} 픽셀")
    print(f"  - Test: {len(df_test):,} 픽셀")
    
    # 대표 패치 선택
    print("\n[2/6] 클래스별 대표 패치 선택 중...")
    selected_patches = select_representative_patches(df_train, df_val, df_test, 
                                                     patches_per_class=2)
    print(f"  - 선택된 클래스 수: {len(selected_patches)}")
    
    # 패치 시각화
    print("\n[3/6] 패치 이미지 생성 중...")
    patch_images = {}
    
    for class_name, patches in selected_patches.items():
        print(f"  - {class_name}: {len(patches)}개 패치")
        for patch_info in patches:
            patch_path = get_patch_path(patch_info)
            if patch_path and patch_path.exists():
                try:
                    image, class_mask, conf_mask = load_patch(patch_path)
                    patch_id = patch_info['patch_id']
                    save_path = output_dir / f"patch_{patch_id.replace('-', '_').replace('_', '_')}.png"
                    save_path = output_dir / f"patch_{class_name.replace(' ', '_')}_{patch_info['image']}.png"
                    
                    visualize_patch(image, class_mask, conf_mask, 
                                  title=f"{class_name} - {patch_id}",
                                  save_path=save_path)
                    patch_images[patch_id] = str(save_path)
                except Exception as e:
                    print(f"    경고: {patch_path} 로드 실패 - {e}")
    
    # 히스토그램 생성
    print("\n[4/6] 히스토그램 생성 중...")
    create_histograms(df_train, df_val, df_test, output_dir)
    
    # 스펙트럴 시그니처 플롯
    print("\n[5/6] 스펙트럴 시그니처 플롯 생성 중...")
    create_spectral_signature_plot(df_train, df_val, df_test, output_dir)
    
    # 마크다운 리포트 생성
    print("\n[6/6] 마크다운 리포트 생성 중...")
    report_path = root_path / 'data' / 'eda_report.md'
    generate_markdown_report(df_train, df_val, df_test, selected_patches,
                            patch_images, report_path)
    
    print(f"\n완료! 리포트가 생성되었습니다: {report_path}")
    print(f"이미지 디렉토리: {output_dir}")

if __name__ == "__main__":
    main()

