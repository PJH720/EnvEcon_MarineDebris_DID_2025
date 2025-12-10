# -*- coding: utf-8 -*-
'''
정책 효과 측정을 위한 시계열 분석
해양 쓰래기 탐지 데이터를 활용하여 정책 효과를 측정하고 시각화합니다.
'''

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
from scipy import stats
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
roi_mapping = assets.roi_mapping

def parse_date_from_filename(filename):
    """파일명에서 날짜 추출"""
    parts = filename.split('_')
    if len(parts) >= 2:
        date_str = parts[1] if parts[0] == 'S2' else parts[0]
        try:
            day, month, year = date_str.split('-')
            year = int(year)
            if year < 50:
                year += 2000
            else:
                year += 1900
            return datetime(int(year), int(month), int(day))
        except:
            return None
    return None

def extract_tile_from_filename(filename):
    """파일명에서 타일 코드 추출"""
    parts = filename.split('_')
    if len(parts) >= 3:
        return parts[2] if parts[0] == 'S2' else parts[1]
    return None

def collect_time_series_data(data_path, patches_path, split_file='test_X.txt'):
    """시계열 데이터 수집"""
    split_path = os.path.join(data_path, 'splits', split_file)
    if not os.path.exists(split_path):
        return None
    
    rois = np.genfromtxt(split_path, dtype='str')
    
    time_series_data = []
    
    print("시계열 데이터 수집 중...")
    for roi in tqdm(rois):
        roi_folder = '_'.join(['S2'] + roi.split('_')[:-1])
        roi_name = '_'.join(['S2'] + roi.split('_'))
        mask_file = os.path.join(patches_path, roi_folder, roi_name + '_cl.tif')
        
        if not os.path.exists(mask_file):
            continue
        
        try:
            with rasterio.open(mask_file) as src:
                mask = src.read(1)
            
            date = parse_date_from_filename(roi)
            tile = extract_tile_from_filename(roi)
            
            if date is None or tile is None:
                continue
            
            total_pixels = np.sum(mask > 0)
            debris_pixels = np.sum(mask == 1)
            
            if total_pixels > 0:
                time_series_data.append({
                    'date': date,
                    'tile': tile,
                    'region': roi_mapping.get(tile, tile),
                    'debris_ratio': debris_pixels / total_pixels,
                    'debris_pixels': debris_pixels,
                    'year': date.year,
                    'month': date.month,
                    'quarter': (date.month - 1) // 3 + 1
                })
        except:
            continue
    
    if not time_series_data:
        return None
    
    df = pd.DataFrame(time_series_data)
    df = df.sort_values('date')
    return df

def define_policy_periods(df):
    """정책 시행 기간 정의 (예시: 사용자가 수정 가능)"""
    # 실제 정책 시행 기간에 맞게 수정 필요
    # 여기서는 데이터 기간을 기반으로 가상의 정책 기간 설정
    
    min_date = df['date'].min()
    max_date = df['date'].max()
    date_range = (max_date - min_date).days
    
    # 예시: 중간 시점을 정책 시행 시점으로 가정
    policy_date = min_date + timedelta(days=date_range // 2)
    
    periods = {
        'pre_policy': (min_date, policy_date - timedelta(days=1)),
        'post_policy': (policy_date, max_date)
    }
    
    return periods, policy_date

def analyze_policy_impact(df, periods, output_dir):
    """정책 효과 분석"""
    os.makedirs(output_dir, exist_ok=True)
    
    pre_start, pre_end = periods['pre_policy']
    post_start, post_end = periods['post_policy']
    
    df_pre = df[(df['date'] >= pre_start) & (df['date'] <= pre_end)]
    df_post = df[(df['date'] >= post_start) & (df['date'] <= post_end)]
    
    # 전체 통계
    pre_mean = df_pre['debris_ratio'].mean()
    post_mean = df_post['debris_ratio'].mean()
    change = post_mean - pre_mean
    change_pct = (change / pre_mean * 100) if pre_mean > 0 else 0
    
    # 통계적 유의성 검정
    t_stat, p_value = stats.ttest_ind(df_pre['debris_ratio'], df_post['debris_ratio'])
    
    # 지역별 분석
    regional_impact = []
    for region in df['region'].unique():
        region_pre = df_pre[df_pre['region'] == region]['debris_ratio']
        region_post = df_post[df_post['region'] == region]['debris_ratio']
        
        if len(region_pre) > 0 and len(region_post) > 0:
            regional_impact.append({
                'region': region,
                'pre_mean': region_pre.mean(),
                'post_mean': region_post.mean(),
                'change': region_post.mean() - region_pre.mean(),
                'change_pct': ((region_post.mean() - region_pre.mean()) / region_pre.mean() * 100) 
                              if region_pre.mean() > 0 else 0,
                'pre_count': len(region_pre),
                'post_count': len(region_post)
            })
    
    regional_impact_df = pd.DataFrame(regional_impact)
    regional_impact_df = regional_impact_df.sort_values('change_pct')
    
    # 시각화
    fig, axes = plt.subplots(2, 2, figsize=(20, 14))
    
    # 1. 정책 전후 비교 (전체)
    ax = axes[0, 0]
    data_to_plot = [df_pre['debris_ratio'], df_post['debris_ratio']]
    bp = ax.boxplot(data_to_plot, labels=['정책 전', '정책 후'], patch_artist=True)
    bp['boxes'][0].set_facecolor('lightcoral')
    bp['boxes'][1].set_facecolor('lightgreen')
    ax.set_ylabel('Marine Debris Ratio', fontsize=12, fontweight='bold')
    ax.set_title(f'Policy Impact: Before vs After\n'
                 f'Change: {change:.4f} ({change_pct:+.2f}%), p-value: {p_value:.4f}',
                 fontsize=14, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    # 평균선 표시
    ax.axhline(pre_mean, color='red', linestyle='--', alpha=0.7, label=f'Pre-mean: {pre_mean:.4f}')
    ax.axhline(post_mean, color='green', linestyle='--', alpha=0.7, label=f'Post-mean: {post_mean:.4f}')
    ax.legend(fontsize=10)
    
    # 2. 시계열 추이 (정책 시점 표시)
    ax = axes[0, 1]
    policy_date = periods['post_policy'][0]
    
    # 이동평균 계산
    window = 30  # 30일 이동평균
    df_sorted = df.sort_values('date')
    df_sorted['ma'] = df_sorted['debris_ratio'].rolling(window=window, center=True).mean()
    
    ax.plot(df_sorted['date'], df_sorted['debris_ratio'], 
           alpha=0.3, color='gray', label='Daily values')
    ax.plot(df_sorted['date'], df_sorted['ma'], 
           color='blue', linewidth=2, label=f'{window}-day moving average')
    ax.axvline(policy_date, color='red', linestyle='--', linewidth=2, 
              label='Policy Implementation')
    ax.fill_between([periods['pre_policy'][0], periods['pre_policy'][1]], 
                    ax.get_ylim()[0], ax.get_ylim()[1], 
                    alpha=0.2, color='red', label='Pre-policy period')
    ax.fill_between([periods['post_policy'][0], periods['post_policy'][1]], 
                    ax.get_ylim()[0], ax.get_ylim()[1], 
                    alpha=0.2, color='green', label='Post-policy period')
    ax.set_ylabel('Marine Debris Ratio', fontsize=12, fontweight='bold')
    ax.set_xlabel('Date', fontsize=12, fontweight='bold')
    ax.set_title('Time Series: Policy Impact Over Time', fontsize=14, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    # 3. 지역별 정책 효과
    ax = axes[1, 0]
    colors = ['green' if x < 0 else 'red' for x in regional_impact_df['change_pct']]
    bars = ax.barh(range(len(regional_impact_df)), regional_impact_df['change_pct'], color=colors, alpha=0.7)
    ax.set_yticks(range(len(regional_impact_df)))
    ax.set_yticklabels(regional_impact_df['region'], fontsize=9)
    ax.set_xlabel('Change in Marine Debris Ratio (%)', fontsize=12, fontweight='bold')
    ax.set_title('Regional Policy Impact\n(% Change by Region)', fontsize=14, fontweight='bold')
    ax.axvline(0, color='black', linestyle='-', linewidth=1)
    ax.grid(axis='x', alpha=0.3)
    # 값 표시
    for i, (idx, row) in enumerate(regional_impact_df.iterrows()):
        ax.text(row['change_pct'], i, f" {row['change_pct']:+.2f}%", 
               va='center', fontsize=9)
    
    # 4. 정책 전후 분포 비교 (히스토그램)
    ax = axes[1, 1]
    ax.hist(df_pre['debris_ratio'], bins=30, alpha=0.6, color='red', 
           label='Pre-policy', density=True)
    ax.hist(df_post['debris_ratio'], bins=30, alpha=0.6, color='green', 
           label='Post-policy', density=True)
    ax.axvline(pre_mean, color='red', linestyle='--', linewidth=2, 
              label=f'Pre-mean: {pre_mean:.4f}')
    ax.axvline(post_mean, color='green', linestyle='--', linewidth=2, 
              label=f'Post-mean: {post_mean:.4f}')
    ax.set_xlabel('Marine Debris Ratio', fontsize=12, fontweight='bold')
    ax.set_ylabel('Density', fontsize=12, fontweight='bold')
    ax.set_title('Distribution Comparison: Before vs After Policy', 
                 fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, 'policy_impact_analysis.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    # 지역별 효과 저장
    impact_path = os.path.join(output_dir, 'regional_policy_impact.csv')
    regional_impact_df.to_csv(impact_path, index=False, encoding='utf-8-sig')
    
    print(f"정책 효과 분석 시각화 저장: {output_path}")
    print(f"지역별 효과 데이터 저장: {impact_path}")
    
    return {
        'pre_mean': pre_mean,
        'post_mean': post_mean,
        'change': change,
        'change_pct': change_pct,
        'p_value': p_value,
        't_statistic': t_stat,
        'regional_impact': regional_impact_df
    }

def visualize_seasonal_adjustment(df, periods, output_dir):
    """계절성 조정 분석"""
    os.makedirs(output_dir, exist_ok=True)
    
    # 월별 평균 계산 (계절성 제거)
    monthly_avg = df.groupby('month')['debris_ratio'].mean()
    
    # 계절성 조정된 데이터
    df['seasonal_adjusted'] = df.apply(
        lambda row: row['debris_ratio'] - monthly_avg[row['month']] + df['debris_ratio'].mean(),
        axis=1
    )
    
    pre_start, pre_end = periods['pre_policy']
    post_start, post_end = periods['post_policy']
    
    df_pre = df[(df['date'] >= pre_start) & (df['date'] <= pre_end)]
    df_post = df[(df['date'] >= post_start) & (df['date'] <= post_end)]
    
    fig, axes = plt.subplots(1, 2, figsize=(18, 6))
    
    # 원본 데이터
    ax = axes[0]
    df_sorted = df.sort_values('date')
    ax.plot(df_sorted['date'], df_sorted['debris_ratio'], 
           alpha=0.5, color='gray', label='Original')
    policy_date = periods['post_policy'][0]
    ax.axvline(policy_date, color='red', linestyle='--', linewidth=2)
    ax.set_ylabel('Marine Debris Ratio', fontsize=12, fontweight='bold')
    ax.set_xlabel('Date', fontsize=12, fontweight='bold')
    ax.set_title('Original Time Series (with Seasonality)', fontsize=14, fontweight='bold')
    ax.grid(alpha=0.3)
    ax.legend()
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    # 계절성 조정된 데이터
    ax = axes[1]
    ax.plot(df_sorted['date'], df_sorted['seasonal_adjusted'], 
           alpha=0.5, color='blue', label='Seasonally Adjusted')
    ax.axvline(policy_date, color='red', linestyle='--', linewidth=2)
    ax.set_ylabel('Seasonally Adjusted Marine Debris Ratio', fontsize=12, fontweight='bold')
    ax.set_xlabel('Date', fontsize=12, fontweight='bold')
    ax.set_title('Seasonally Adjusted Time Series', fontsize=14, fontweight='bold')
    ax.grid(alpha=0.3)
    ax.legend()
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, 'seasonal_adjustment.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"계절성 조정 분석 저장: {output_path}")

def main():
    """메인 함수"""
    # 경로 설정
    data_path = os.path.join(root_path, 'data')
    patches_path = os.path.join(data_path, 'patches')
    output_dir = os.path.join(data_path, 'presentation', 'policy_impact')
    
    os.makedirs(output_dir, exist_ok=True)
    
    print("=" * 80)
    print("정책 효과 측정을 위한 시계열 분석")
    print("=" * 80)
    
    # 시계열 데이터 수집
    df = collect_time_series_data(data_path, patches_path, 'test_X.txt')
    
    if df is None or len(df) == 0:
        print("경고: 분석할 데이터가 없습니다.")
        return
    
    print(f"\n총 {len(df)}개 샘플 분석")
    print(f"기간: {df['date'].min()} ~ {df['date'].max()}")
    
    # 정책 기간 정의
    periods, policy_date = define_policy_periods(df)
    print(f"\n정책 시행 시점 (가정): {policy_date}")
    print(f"정책 전 기간: {periods['pre_policy'][0]} ~ {periods['pre_policy'][1]}")
    print(f"정책 후 기간: {periods['post_policy'][0]} ~ {periods['post_policy'][1]}")
    
    # 정책 효과 분석
    print("\n정책 효과 분석 중...")
    impact_results = analyze_policy_impact(df, periods, output_dir)
    
    # 계절성 조정 분석
    print("\n계절성 조정 분석 중...")
    visualize_seasonal_adjustment(df, periods, output_dir)
    
    # 요약 리포트 생성
    summary_path = os.path.join(output_dir, 'policy_impact_summary.txt')
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("정책 효과 측정 결과 요약\n")
        f.write("=" * 80 + "\n\n")
        
        f.write(f"분석 기간: {df['date'].min()} ~ {df['date'].max()}\n")
        f.write(f"정책 시행 시점: {policy_date}\n\n")
        
        f.write("전체 정책 효과:\n")
        f.write("-" * 80 + "\n")
        f.write(f"정책 전 평균: {impact_results['pre_mean']:.4f}\n")
        f.write(f"정책 후 평균: {impact_results['post_mean']:.4f}\n")
        f.write(f"변화량: {impact_results['change']:.4f} ({impact_results['change_pct']:+.2f}%)\n")
        f.write(f"t-통계량: {impact_results['t_statistic']:.4f}\n")
        f.write(f"p-value: {impact_results['p_value']:.4f}\n")
        
        if impact_results['p_value'] < 0.05:
            f.write("결과: 통계적으로 유의한 변화 (p < 0.05)\n")
        else:
            f.write("결과: 통계적으로 유의하지 않은 변화 (p >= 0.05)\n")
        
        f.write("\n지역별 정책 효과 (상위 5개):\n")
        f.write("-" * 80 + "\n")
        top_impact = impact_results['regional_impact'].nlargest(5, 'change_pct')
        for idx, row in top_impact.iterrows():
            f.write(f"{row['region']:30s}: {row['change_pct']:+.2f}%\n")
    
    print(f"\n분석 완료! 결과는 {output_dir}에 저장되었습니다.")
    print(f"요약 리포트: {summary_path}")

if __name__ == "__main__":
    main()

