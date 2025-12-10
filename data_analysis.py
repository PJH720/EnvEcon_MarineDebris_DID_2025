# -*- coding: utf-8 -*-
'''
데이터 분석 스크립트
dataset.h5 파일을 사용하여 MARIDA 데이터셋의 스펙트럴 분석을 수행합니다.
'''

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# seaborn이 없어도 작동하도록
try:
    import seaborn as sns
    HAS_SEABORN = True
except ImportError:
    HAS_SEABORN = False

# 프로젝트 루트 경로 추가
root_path = Path(__file__).parent
sys.path.append(str(root_path / 'utils'))
from assets import cat_mapping, labels, color_mapping, s2_mapping

# 한글 폰트 설정 (Windows)
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

def load_data():
    """데이터셋 로드"""
    print("=" * 60)
    print("데이터셋 로드 중...")
    print("=" * 60)
    
    hdf_path = root_path / 'data' / 'dataset.h5'
    
    if not hdf_path.exists():
        raise FileNotFoundError(f"dataset.h5 파일을 찾을 수 없습니다: {hdf_path}")
    
    hdf = pd.HDFStore(str(hdf_path), mode='r')
    
    df_train = hdf.select('train')
    df_val = hdf.select('val')
    df_test = hdf.select('test')
    
    hdf.close()
    
    print(f"[OK] Train 데이터: {len(df_train):,} 픽셀")
    print(f"[OK] Val 데이터: {len(df_val):,} 픽셀")
    print(f"[OK] Test 데이터: {len(df_test):,} 픽셀")
    print(f"[OK] 전체 데이터: {len(df_train) + len(df_val) + len(df_test):,} 픽셀")
    
    return df_train, df_val, df_test

def basic_statistics(df_train, df_val, df_test):
    """기본 통계 정보"""
    print("\n" + "=" * 60)
    print("기본 통계 정보")
    print("=" * 60)
    
    df_all = pd.concat([df_train, df_val, df_test], ignore_index=True)
    
    print(f"\n컬럼 정보:")
    print(df_all.columns.tolist())
    
    print(f"\n데이터 타입:")
    print(df_all.dtypes)
    
    print(f"\n결측값:")
    print(df_all.isnull().sum())
    
    print(f"\n스펙트럴 밴드 통계:")
    band_columns = [col for col in df_all.columns if col.startswith('nm')]
    print(df_all[band_columns].describe())

def class_distribution(df_train, df_val, df_test):
    """클래스별 분포 분석"""
    print("\n" + "=" * 60)
    print("클래스별 분포 분석")
    print("=" * 60)
    
    df_all = pd.concat([df_train, df_val, df_test], ignore_index=True)
    
    # 클래스별 픽셀 수
    class_counts = df_all['Class'].value_counts().sort_index()
    
    print("\n클래스별 픽셀 수:")
    for class_name, count in class_counts.items():
        percentage = (count / len(df_all)) * 100
        print(f"  {class_name:25s}: {count:10,} ({percentage:5.2f}%)")
    
    # Split별 클래스 분포
    print("\nSplit별 클래스 분포:")
    for split_name, df in [('Train', df_train), ('Val', df_val), ('Test', df_test)]:
        print(f"\n{split_name}:")
        split_counts = df['Class'].value_counts().sort_index()
        for class_name, count in split_counts.items():
            percentage = (count / len(df)) * 100
            print(f"  {class_name:25s}: {count:10,} ({percentage:5.2f}%)")
    
    return class_counts

def confidence_distribution(df_train, df_val, df_test):
    """신뢰도 레벨별 분포"""
    print("\n" + "=" * 60)
    print("신뢰도 레벨별 분포")
    print("=" * 60)
    
    df_all = pd.concat([df_train, df_val, df_test], ignore_index=True)
    
    conf_counts = df_all['Confidence'].value_counts()
    
    print("\n신뢰도 레벨별 픽셀 수:")
    for conf, count in conf_counts.items():
        percentage = (count / len(df_all)) * 100
        print(f"  {conf:15s}: {count:10,} ({percentage:5.2f}%)")
    
    return conf_counts

def spectral_signature_analysis(df_train, df_val, df_test):
    """스펙트럴 시그니처 분석"""
    print("\n" + "=" * 60)
    print("클래스별 평균 스펙트럴 시그니처")
    print("=" * 60)
    
    df_all = pd.concat([df_train, df_val, df_test], ignore_index=True)
    
    band_columns = sorted([col for col in df_all.columns if col.startswith('nm')])
    wavelengths = [int(col.replace('nm', '')) for col in band_columns]
    
    # 클래스별 평균 스펙트럴 시그니처
    class_spectra = df_all.groupby('Class')[band_columns].mean()
    
    print("\n주요 클래스별 평균 반사율:")
    for class_name in ['Marine Debris', 'Dense Sargassum', 'Marine Water', 'Clouds']:
        if class_name in class_spectra.index:
            print(f"\n{class_name}:")
            for band, wavelength in zip(band_columns, wavelengths):
                value = class_spectra.loc[class_name, band]
                print(f"  {band:8s} ({wavelength:4d}nm): {value:8.4f}")
    
    return class_spectra, wavelengths, band_columns

def create_visualizations(df_train, df_val, df_test, class_counts, conf_counts, 
                         class_spectra, wavelengths, band_columns):
    """시각화 생성"""
    print("\n" + "=" * 60)
    print("시각화 생성 중...")
    print("=" * 60)
    
    # 출력 디렉토리 생성
    output_dir = root_path / 'data' / 'analysis_results'
    output_dir.mkdir(exist_ok=True)
    
    df_all = pd.concat([df_train, df_val, df_test], ignore_index=True)
    
    # 1. 클래스별 분포 막대 그래프
    plt.figure(figsize=(14, 8))
    colors = [color_mapping.get(class_name, 'gray') for class_name in class_counts.index]
    plt.barh(range(len(class_counts)), class_counts.values, color=colors)
    plt.yticks(range(len(class_counts)), class_counts.index)
    plt.xlabel('픽셀 수', fontsize=12)
    plt.title('클래스별 픽셀 분포', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / 'class_distribution.png', dpi=300, bbox_inches='tight')
    print(f"[OK] 저장됨: {output_dir / 'class_distribution.png'}")
    plt.close()
    
    # 2. 신뢰도 레벨별 분포
    plt.figure(figsize=(10, 6))
    plt.bar(conf_counts.index, conf_counts.values, color=['green', 'orange', 'red'])
    plt.xlabel('신뢰도 레벨', fontsize=12)
    plt.ylabel('픽셀 수', fontsize=12)
    plt.title('신뢰도 레벨별 분포', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / 'confidence_distribution.png', dpi=300, bbox_inches='tight')
    print(f"[OK] 저장됨: {output_dir / 'confidence_distribution.png'}")
    plt.close()
    
    # 3. 주요 클래스별 스펙트럴 시그니처
    plt.figure(figsize=(12, 8))
    main_classes = ['Marine Debris', 'Dense Sargassum', 'Sparse Sargassum', 
                    'Marine Water', 'Clouds', 'Ship']
    
    for class_name in main_classes:
        if class_name in class_spectra.index:
            spectrum = class_spectra.loc[class_name, band_columns].values
            color = color_mapping.get(class_name, 'gray')
            plt.plot(wavelengths, spectrum, marker='o', label=class_name, 
                    color=color, linewidth=2, markersize=6)
    
    plt.xlabel('파장 (nm)', fontsize=12)
    plt.ylabel('평균 반사율', fontsize=12)
    plt.title('주요 클래스별 평균 스펙트럴 시그니처', fontsize=14, fontweight='bold')
    plt.legend(loc='best', fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / 'spectral_signatures.png', dpi=300, bbox_inches='tight')
    print(f"[OK] 저장됨: {output_dir / 'spectral_signatures.png'}")
    plt.close()
    
    # 4. Split별 클래스 분포 히트맵
    split_class_counts = pd.DataFrame({
        'Train': df_train['Class'].value_counts(),
        'Val': df_val['Class'].value_counts(),
        'Test': df_test['Class'].value_counts()
    }).fillna(0)
    
    plt.figure(figsize=(10, 12))
    if HAS_SEABORN:
        sns.heatmap(split_class_counts, annot=True, fmt='.0f', cmap='YlOrRd', 
                    cbar_kws={'label': '픽셀 수'}, linewidths=0.5)
    else:
        im = plt.imshow(split_class_counts.values, aspect='auto', cmap='YlOrRd')
        plt.colorbar(im, label='픽셀 수')
        for i in range(len(split_class_counts.index)):
            for j in range(len(split_class_counts.columns)):
                plt.text(j, i, f'{int(split_class_counts.iloc[i, j])}', 
                        ha='center', va='center', color='black', fontsize=8)
        plt.xticks(range(len(split_class_counts.columns)), split_class_counts.columns)
        plt.yticks(range(len(split_class_counts.index)), split_class_counts.index)
    plt.xlabel('Split', fontsize=12)
    plt.ylabel('클래스', fontsize=12)
    plt.title('Split별 클래스 분포', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / 'split_class_heatmap.png', dpi=300, bbox_inches='tight')
    print(f"[OK] 저장됨: {output_dir / 'split_class_heatmap.png'}")
    plt.close()
    
    # 5. 스펙트럴 밴드별 상관관계
    band_columns = sorted([col for col in df_all.columns if col.startswith('nm')])
    correlation_matrix = df_all[band_columns].corr()
    
    plt.figure(figsize=(10, 8))
    if HAS_SEABORN:
        sns.heatmap(correlation_matrix, annot=True, fmt='.2f', cmap='coolwarm', 
                    center=0, square=True, linewidths=0.5, cbar_kws={'label': '상관계수'})
    else:
        im = plt.imshow(correlation_matrix.values, aspect='auto', cmap='coolwarm', 
                       vmin=-1, vmax=1)
        plt.colorbar(im, label='상관계수')
        for i in range(len(correlation_matrix.index)):
            for j in range(len(correlation_matrix.columns)):
                plt.text(j, i, f'{correlation_matrix.iloc[i, j]:.2f}', 
                        ha='center', va='center', color='black', fontsize=7)
        plt.xticks(range(len(correlation_matrix.columns)), correlation_matrix.columns, rotation=45)
        plt.yticks(range(len(correlation_matrix.index)), correlation_matrix.index)
    plt.title('스펙트럴 밴드 간 상관관계', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / 'band_correlation.png', dpi=300, bbox_inches='tight')
    print(f"[OK] 저장됨: {output_dir / 'band_correlation.png'}")
    plt.close()

def save_summary_report(df_train, df_val, df_test, class_counts, conf_counts):
    """요약 보고서 저장"""
    output_dir = root_path / 'data' / 'analysis_results'
    output_dir.mkdir(exist_ok=True)
    
    report_path = output_dir / 'analysis_summary.txt'
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("=" * 60 + "\n")
        f.write("MARIDA 데이터셋 분석 요약 보고서\n")
        f.write("=" * 60 + "\n\n")
        
        df_all = pd.concat([df_train, df_val, df_test], ignore_index=True)
        
        f.write(f"전체 데이터 통계:\n")
        f.write(f"  - 총 픽셀 수: {len(df_all):,}\n")
        f.write(f"  - Train: {len(df_train):,} ({len(df_train)/len(df_all)*100:.2f}%)\n")
        f.write(f"  - Val: {len(df_val):,} ({len(df_val)/len(df_all)*100:.2f}%)\n")
        f.write(f"  - Test: {len(df_test):,} ({len(df_test)/len(df_all)*100:.2f}%)\n\n")
        
        f.write(f"클래스별 분포:\n")
        for class_name, count in class_counts.items():
            percentage = (count / len(df_all)) * 100
            f.write(f"  - {class_name:25s}: {count:10,} ({percentage:5.2f}%)\n")
        
        f.write(f"\n신뢰도 레벨별 분포:\n")
        for conf, count in conf_counts.items():
            percentage = (count / len(df_all)) * 100
            f.write(f"  - {conf:15s}: {count:10,} ({percentage:5.2f}%)\n")
    
    print(f"[OK] 저장됨: {report_path}")

def main():
    """메인 함수"""
    print("\n" + "=" * 60)
    print("MARIDA 데이터셋 스펙트럴 분석")
    print("=" * 60 + "\n")
    
    try:
        # 데이터 로드
        df_train, df_val, df_test = load_data()
        
        # 기본 통계
        basic_statistics(df_train, df_val, df_test)
        
        # 클래스별 분포
        class_counts = class_distribution(df_train, df_val, df_test)
        
        # 신뢰도 분포
        conf_counts = confidence_distribution(df_train, df_val, df_test)
        
        # 스펙트럴 시그니처 분석
        class_spectra, wavelengths, band_columns = spectral_signature_analysis(
            df_train, df_val, df_test)
        
        # 시각화
        create_visualizations(df_train, df_val, df_test, class_counts, conf_counts,
                            class_spectra, wavelengths, band_columns)
        
        # 요약 보고서 저장
        save_summary_report(df_train, df_val, df_test, class_counts, conf_counts)
        
        print("\n" + "=" * 60)
        print("분석 완료!")
        print("=" * 60)
        print(f"\n결과는 'data/analysis_results/' 디렉토리에 저장되었습니다.")
        
    except Exception as e:
        print(f"\n오류 발생: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()

