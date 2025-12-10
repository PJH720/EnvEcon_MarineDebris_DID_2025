# MARIDA 데이터셋 탐색적 데이터 분석 (EDA) 리포트

## 1. 데이터셋 개요

- **총 픽셀 수**: 837,377
- **Train**: 429,412 (51.28%)
- **Val**: 213,102 (25.45%)
- **Test**: 194,863 (23.27%)

### 1.1 스펙트럴 밴드 정보

Sentinel-2 밴드:
- nm1600: 1600nm
- nm2200: 2200nm
- nm440: 440nm
- nm490: 490nm
- nm560: 560nm
- nm665: 665nm
- nm705: 705nm
- nm740: 740nm
- nm783: 783nm
- nm842: 842nm
- nm865: 865nm

## 2. 클래스별 통계

### 2.1 클래스별 픽셀 분포

| 클래스 | 픽셀 수 | 비율 (%) |
|--------|---------|----------|
| Sediment-Laden Water | 372,937 | 44.54 |
| Turbid Water | 157,612 | 18.82 |
| Marine Water | 129,159 | 15.42 |
| Clouds | 117,400 | 14.02 |
| Shallow Water | 17,369 | 2.07 |
| Cloud Shadows | 11,728 | 1.40 |
| Wakes | 8,490 | 1.01 |
| Waves | 5,827 | 0.70 |
| Ship | 5,803 | 0.69 |
| Marine Debris | 3,399 | 0.41 |
| Dense Sargassum | 2,797 | 0.33 |
| Sparse Sargassum | 2,357 | 0.28 |
| Foam | 1,225 | 0.15 |
| Natural Organic Material | 864 | 0.10 |
| Mixed Water | 410 | 0.05 |

### 2.2 Split별 클래스 분포

| 클래스 | Train | Val | Test |
|--------|-------|-----|------|
| Sediment-Laden Water | 154,335 | 125,565 | 93,037 |
| Turbid Water | 86,820 | 38,566 | 32,226 |
| Marine Water | 86,877 | 18,839 | 23,443 |
| Clouds | 65,295 | 19,262 | 32,843 |
| Shallow Water | 13,852 | 1,011 | 2,506 |
| Cloud Shadows | 5,675 | 2,404 | 3,649 |
| Wakes | 4,974 | 1,946 | 1,570 |
| Waves | 2,975 | 987 | 1,865 |
| Ship | 3,289 | 1,340 | 1,174 |
| Marine Debris | 1,943 | 1,075 | 381 |
| Dense Sargassum | 870 | 1,167 | 760 |
| Sparse Sargassum | 1,091 | 385 | 881 |
| Foam | 469 | 369 | 387 |
| Natural Organic Material | 723 | 92 | 49 |
| Mixed Water | 224 | 94 | 92 |

## 3. 대표 패치 시각화

### 3.1 Sediment-Laden Water

#### 패치 1: 23-9-20_16PCC_7

- **픽셀 수**: 50,303
- **날짜**: 23-9-20
- **타일**: 16PCC

![23-9-20_16PCC_7](eda_images/patch_Sediment-Laden_Water_7.png)

#### 패치 2: 4-9-16_16PCC_12

- **픽셀 수**: 32,673
- **날짜**: 4-9-16
- **타일**: 16PCC

![4-9-16_16PCC_12](eda_images/patch_Sediment-Laden_Water_12.png)

### 3.2 Turbid Water

#### 패치 1: 23-9-20_16PCC_11

- **픽셀 수**: 23,053
- **날짜**: 23-9-20
- **타일**: 16PCC

![23-9-20_16PCC_11](eda_images/patch_Turbid_Water_11.png)

#### 패치 2: 4-9-16_16PCC_22

- **픽셀 수**: 16,578
- **날짜**: 4-9-16
- **타일**: 16PCC

![4-9-16_16PCC_22](eda_images/patch_Turbid_Water_22.png)

### 3.3 Marine Water

#### 패치 1: 20-4-18_30VWH_9

- **픽셀 수**: 21,207
- **날짜**: 20-4-18
- **타일**: 30VWH

![20-4-18_30VWH_9](eda_images/patch_Marine_Water_9.png)

#### 패치 2: 13-12-18_16PCC_18

- **픽셀 수**: 7,415
- **날짜**: 13-12-18
- **타일**: 16PCC

![13-12-18_16PCC_18](eda_images/patch_Marine_Water_18.png)

### 3.4 Clouds

#### 패치 1: 3-11-16_16PDC_7

- **픽셀 수**: 6,617
- **날짜**: 3-11-16
- **타일**: 16PDC

![3-11-16_16PDC_7](eda_images/patch_Clouds_7.png)

#### 패치 2: 4-9-19_16PCC_10

- **픽셀 수**: 5,470
- **날짜**: 4-9-19
- **타일**: 16PCC

![4-9-19_16PCC_10](eda_images/patch_Clouds_10.png)

### 3.5 Shallow Water

#### 패치 1: 8-3-18_16QED_6

- **픽셀 수**: 4,186
- **날짜**: 8-3-18
- **타일**: 16QED

![8-3-18_16QED_6](eda_images/patch_Shallow_Water_6.png)

#### 패치 2: 9-10-17_16PEC_12

- **픽셀 수**: 2,327
- **날짜**: 9-10-17
- **타일**: 16PEC

![9-10-17_16PEC_12](eda_images/patch_Shallow_Water_12.png)

### 3.6 Cloud Shadows

#### 패치 1: 20-4-18_30VWH_2

- **픽셀 수**: 1,312
- **날짜**: 20-4-18
- **타일**: 30VWH

![20-4-18_30VWH_2](eda_images/patch_Cloud_Shadows_2.png)

#### 패치 2: 27-1-19_16QED_7

- **픽셀 수**: 945
- **날짜**: 27-1-19
- **타일**: 16QED

![27-1-19_16QED_7](eda_images/patch_Cloud_Shadows_7.png)

### 3.7 Wakes

#### 패치 1: 24-8-20_16PCC_23

- **픽셀 수**: 1,246
- **날짜**: 24-8-20
- **타일**: 16PCC

![24-8-20_16PCC_23](eda_images/patch_Wakes_23.png)

#### 패치 2: 24-8-20_16PCC_25

- **픽셀 수**: 844
- **날짜**: 24-8-20
- **타일**: 16PCC

![24-8-20_16PCC_25](eda_images/patch_Wakes_25.png)

### 3.8 Waves

#### 패치 1: 22-3-20_18QWF_3

- **픽셀 수**: 520
- **날짜**: 22-3-20
- **타일**: 18QWF

![22-3-20_18QWF_3](eda_images/patch_Waves_3.png)

#### 패치 2: 22-3-20_18QWF_5

- **픽셀 수**: 489
- **날짜**: 22-3-20
- **타일**: 18QWF

![22-3-20_18QWF_5](eda_images/patch_Waves_5.png)

### 3.9 Ship

#### 패치 1: 14-9-18_16PCC_39

- **픽셀 수**: 165
- **날짜**: 14-9-18
- **타일**: 16PCC

![14-9-18_16PCC_39](eda_images/patch_Ship_39.png)

#### 패치 2: 18-9-20_16PCC_68

- **픽셀 수**: 163
- **날짜**: 18-9-20
- **타일**: 16PCC

![18-9-20_16PCC_68](eda_images/patch_Ship_68.png)

### 3.10 Marine Debris

#### 패치 1: 4-9-16_16PCC_20

- **픽셀 수**: 153
- **날짜**: 4-9-16
- **타일**: 16PCC

![4-9-16_16PCC_20](eda_images/patch_Marine_Debris_20.png)

#### 패치 2: 4-9-16_16PCC_24

- **픽셀 수**: 143
- **날짜**: 4-9-16
- **타일**: 16PCC

![4-9-16_16PCC_24](eda_images/patch_Marine_Debris_24.png)


## 4. 스펙트럴 밴드 히스토그램

### 4.1 전체 데이터셋 밴드별 히스토그램

![All Bands Histogram](eda_images/histograms_all_bands.png)

### 4.2 클래스별 밴드별 히스토그램

![Class-wise Histogram](eda_images/histograms_by_class.png)

## 5. 클래스별 스펙트럴 시그니처

![Spectral Signatures](eda_images/spectral_signatures.png)

## 6. 특이 밴드 분석

### 6.1 클래스별 평균 반사율 (상위 3개 밴드)

| 클래스 | 1위 밴드 | 2위 밴드 | 3위 밴드 |
|--------|----------|----------|----------|
| Sediment-Laden Water | nm665 (0.1356) | nm705 (0.1239) | nm783 (0.1088) |
| Turbid Water | nm560 (0.0558) | nm490 (0.0492) | nm440 (0.0454) |
| Marine Water | nm440 (0.0386) | nm490 (0.0335) | nm560 (0.0243) |
| Clouds | nm440 (0.1695) | nm490 (0.1660) | nm865 (0.1613) |
| Shallow Water | nm490 (0.0583) | nm440 (0.0565) | nm560 (0.0532) |
| Cloud Shadows | nm440 (0.0396) | nm490 (0.0320) | nm560 (0.0232) |
| Wakes | nm490 (0.0505) | nm440 (0.0484) | nm560 (0.0431) |
| Waves | nm490 (0.0516) | nm440 (0.0511) | nm560 (0.0416) |
| Ship | nm665 (0.1156) | nm490 (0.1084) | nm865 (0.1082) |
| Marine Debris | nm490 (0.0623) | nm842 (0.0575) | nm665 (0.0568) |

---

*이 리포트는 자동으로 생성되었습니다.*
