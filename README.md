## 📄 README.md: 해양 잔해 감소 정책 효과 분석 연구 (Motagua River Basin)

### **[2025학년도 2학기 환경경제학(ECO3005)] 최종 그룹 프로젝트 실증 분석 자료**

#### **프로젝트 제목 (Project Title)**
위성 빅데이터 및 이중차분법(DID)을 활용한 Motagua River Basin 해양 잔해 감축 정책의 효과 실증 분석
*(Evaluation of the Effectiveness of Marine Debris Reduction Policy through DID Analysis)*

###### Presentation

- EnvEcon Beamer (GDrive):
[Evaluation of the Effectiveness of Marine Debris Reduction Policy through DID Analysis (PDF)](https://drive.google.com/file/d/1J-W6gtSYffuhDKcKbGKSkkM9lHvE8buB/view?usp=sharing)[^1]

---

#### Course context 

이 프로젝트는 서강대학교 경제학과 **환경경제학(Environmental Economics, ECO3005)** 2025년 2학기 수업의 **그룹 프로젝트**로 수행되었다.

- 과목명: Environmental Economics (ECO3005)
- 학기: 2025 Fall
- 담당 교수: 신혜선 교수(Shin, Hyeseon)

본 저장소는 수업의 **Group Project Guidelines**에 따라,

1) 환경 문제 정의,
2) 데이터 기반 경제 분석,
3) 정책 평가 및 시사점 제시,
4) 참고문헌 정리
까지를 모두 포함하도록 구성 되었습니다.

---

#### **저자 및 소속 (Authors & Affiliation)**
*   재현 박 (Jaehyun Park)\*, *(서강대학교 경제학과)*
*   정종혁 (Jonghyeok Jeong)†, *(서강대학교 경제학과)*


\* First author, † Co-author

---

#### **1. 연구 개요 및 핵심 결과 (Abstract & Key Findings)**

본 연구는 과테말라 **Motagua River Basin**에 시행된 해양 플라스틱 차단 및 수거 정책(2019년 10월 31일 시행)이 해양 잔해 비율에 미친 인과적 효과를 실증적으로 평가했습니다. 유럽우주국(ESA)의 **Sentinel-2** 위성 이미지와 **MARIDA** 데이터셋으로 훈련된 딥러닝 모델을 사용하여 해양 잔해 비율(Marine Debris Ratio)을 정량적 지표로 사용했습니다.

**이중차분법(DID)** 분석을 통해 정책 효과를 추정한 결과는 다음과 같습니다.
- $\beta_3 \approx -0.095$ (감소 효과 방향), 
- p-value ≈ 0.64로 **통계적으로 유의하지 않음**

정책의 효과는 **통계적으로 유의미하지 않았습니다**. 정책 후 Motagua의 잔해 비율은 완만히 증가하는 시계열 패턴을 보여, 관측 기간 내 정책만으로는 해양 잔해 감소 효과를 명확히 확인하기 어렵습니다. 특히 정책 후 대조군 샘플이 매우 적어(DID 가정 및 검정력 문제), 정책 효과 추정에 구조적 한계가 존재합니다.

| 구분                  | 내용                                                                                                                               |
| :------------------ | :------------------------------------------------------------------------------------------------------------------------------- |
| **핵심 연구 질문**        | Motagua 지역 정책이 해양 잔해 비율을 유의하게 감소시켰는가?                                                                                            |
| **핵심 분석 방법**        | Sentinel-2 + MARIDA U-Net, 이중차분법 (Difference-in-Differences, DID)을 통한 인과 효과 추정, Seasonal adjustment (time series data), 단순 상관 분석 |
| **정책 효과 (DID 추정치)** | **정책 효과 (DID 추정치)** : **-0.095363**로 음(-)의 방향이나(감소 효과 추정) <br>p-value>0.6으로 **유의하지 않음**                                          |
| **정책적 해석**          | 데이터·식별 한계로 인해 관측된 기간 내 정책 효과를 확정하기 어려움                                                                                           |

---

#### **2. 환경경제학적 배경 (Environmental Economic Framework)**

본 연구는 해양 오염 문제를 환경경제학의 핵심 개념에 기반하여 정당화했습니다.

1.  **부정적 외부효과 및 시장 실패 (Negative Externality & Market Failure)**
    *   해양 플라스틱 오염은 생산 및 소비 주체가 사회적 피해 비용(한계피해, MD)을 고려하지 않아 발생하는 **부정적 외부효과**입니다.
    *   이는 **사회적 한계비용(SMC)** 이 **사적 한계비용(PMC)** 을 초과하게 하여, 시장이 **사회적 최적 수준($Q^*$)보다 과도한 오염을 유발(과잉 생산)** 하는 **시장 실패**를 초래합니다. 그 결과 **자중손실(Deadweight Loss, DWL)** 이 발생합니다.
2.  **공유지의 비극 (Tragedy of the Commons)**
    *   해양 및 강 생태계는 비배제성 및 경합성을 특징으로 하는 **공개 접근 자원(Open-access Resource)** 입니다.
    *   높은 거래 비용으로 인해 개인들이 협상할 수 없어, 개인의 이익 추구가 자원의 **과잉 사용**과 고갈을 초래하는 **공유지의 비극**이 발생하며, 이는 정부 개입의 강력한 정당성을 제공합니다.
3. **비시장 가치 평가 \& 비용-편익 분석**
    - 플라스틱 오염은 관광, 어업, 주거환경 등 비시장 재화의 가치에도 영향을 미치므로, 수업에서 다루는 비용-편익 분석과 비시장 가치 평가의 전형적인 응용 사례입니다.

---

#### **3. 데이터 및 기술적 접근법 (Data and Technical Approach)**

본 연구는 **Python** 코딩 및 **빅데이터 처리** 능력을 활용하여 복잡한 원격 탐사 데이터를 경제 분석에 적합한 형태로 구축했습니다.

1.  **종속 변수 구축 (Remote Sensing)**
    *   **데이터 출처:** 유럽우주국(ESA)의 **Sentinel-2** 다중분광 위성 이미지.
    *   **학습 데이터:** 해양 잔해 탐지 벤치마크 데이터셋인 **MARIDA (Marine Debris Archive)** 를 사용했습니다. MARIDA는 해조류, 선박, 거품 등 15가지 해양 특징과 플라스틱 잔해를 구분하도록 정교하게 주석 처리된 데이터입니다.
    *   **모델:** Sentinel-2 데이터를 기반으로 **U-Net** 구조를 채택한 심층 분할(Semantic Segmentation) 모델을 활용하여 픽셀 단위로 잔해 비율을 추정했습니다,.
2.  **DID 분석 환경**
    *   **분석 기간:** 2016‑07‑17 ~ 2020‑12‑22 (N=359)
    *   **처치군 (Treatment):** Motagua River Basin (16PCC 타일).
    *   **대조군 (Control):** 기타 모든 지역.
    *   **정책 시점:**  2019‑10‑31 (Guatemala 일회용 플라스틱 규제 및 플라스틱 차단)
    *   DID 모형:

$$
\text{debris\_ratio}_{it} = \beta_0 + \beta_1 \text{Treated}_i + \beta_2 \text{Post}_t + \beta_3 (\text{Treated}_i \times \text{Post}_t) + \epsilon_{it}
$$
    - 추가로, 시계열 상관분석 및 계절성 조정(Seasonal adjustment)으로 정책 전후 동태를 점검한다.

---

#### **4. 실증 분석 결과 요약 및 한계 (Empirical Results & Limitations)**

| 항목 | 처치군 (Motagua) | 대조군 (기타 지역) |
| :--- | :--- | :--- |
| **정책 전 평균** | 0.161439 (n=121) | 0.105041 (n=168) |
| **정책 후 평균** | 0.183842 (n=68) | **0.222807 (n=2)** |
| **변화량** | +0.022403 (증가) | +0.117766 (증가) |
| **DID ($\beta_3$)** | **-0.095363** (p=0.6361) | |

1.  **정책 효과의 불투명성 (Inconclusive Effect)**
    *   DID 추정치는 -0.0954로 감소하는 경향을 보였으나, p-value가 0.6361로 **통계적 유의성 기준($p < 0.05$)을 충족하지 못했습니다**. 이는 정책 효과가 0이라는 **귀무가설을 기각할 수 없음**을 의미합니다.
2.  **데이터 및 방법론적 한계 (Critical Limitations)**
    *   **대조군의 극심한 샘플 부족:** 정책 후 대조군의 샘플 수($n=2$)가 매우 적어, 대조군의 정책 후 평균 변화량($+0.117766$)이 **극도의 변동성 하에서 계산**되었으며, 이는 DID 추정치($\beta_3$)의 정밀도와 신뢰성을 심각하게 훼손하는 **결정적 한계**입니다,,.
    *   **평행 추세 가정 위협:** 처치군의 변화량(+0.0224)과 대조군의 변화량(+0.1178)이 크게 달라, DID 분석의 핵심인 **평행 추세 가정**이 위배되었을 가능성이 있습니다.
    *   **동태적 추이:** Motagua 지역의 잔해 비율은 정책 시행 후 **시간 경과에 따라 유의하게 증가하는 경향**을 보였습니다 ($r=0.2410$, $p=0.0477$),. 이는 정책의 의도와 상반되는 동태적 추이를 시사하며, DID 분석과 함께 추가적인 심층 분석이 필요합니다.

---

#### **5. 정책적 시사점 및 향후 연구 제언 (Policy Implications & Future Work)**

- **계량경제학적 보강 (Causal Inference)**
    1.   **검정력 강화:** 현재의 효과 크기를 입증하려면 유효 표본 크기를 약 **17배** 늘려야 하므로, 장기 패널 데이터를 추가하거나 표본 크기의 불균형을 해소해야 합니다.
    2.   **동태적 분석:** 지역 및 시간 **고정효과(Fixed Effects)**를 포함한 **이벤트 스터디(Event Study)** 설계를 적용하여 평행 추세 가정을 명시적으로 검증하고 정책 효과가 시간에 따라 어떻게 발현되는지 추적해야 합니다.
- 향후 연구 방향 예시는 다음과 같습니다.
	1. **데이터 확장**: 더 긴 기간, 더 많은 타일과 관측치를 포함해 표본 크기와 대조군의 다양성을 확보.
	2. **이벤트 스터디(Event Study)**: 시점별 효과를 추적하고 평행 추세 가정을 검증.
	3. **경제 지표 연계**: 관광, 수산업, 주택가격 등 환경경제학 수업에서 다루는 실질 경제 변수와 연동한 분석 설계.

---

#### **6. 프로젝트 구조 및 재현 가능성 (Project Structure & Reproducibility)**

본 저장소에는 연구 보고서 작성 및 분석 과정에서 사용된 모든 코드, 데이터 요약, 결과물 및 학술 자료가 포함되어 있습니다.

| 디렉토리/파일                                                                                                       | 설명                                        | 사용 도구               |
| :------------------------------------------------------------------------------------------------------------ | :---------------------------------------- | :------------------ |
| `!EnvEcon_Beamer) Evaluation of the Effectiveness of Marine Debris Reduction Policy through DID Analysis.pdf` | **최종 제출된 연구 보고서 원본 파일**                   | Overleaf / BibTeX   |
| `did_analysis_motagua/`                                                                                       | Motagua DID 분석의 시각화 결과 및 통계 요약 (PNG, TXT) | Python / Matplotlib |
| `regional_distribution.png`                                                                                   | 지역별 해양 잔해 비율 및 샘플 수 분포 시각화                | Python              |
| `temporal_distribution.png`                                                                                   | 연도별 및 계절별 해양 잔해 비율 시계열 분석                 | Python              |
| `analysis_summary.txt` / `cost_benefit_summary.md`                                                            | DID 및 탐지 정확도 기반 비용-편익 가상 분석 요약            | Python              |
| `README.md`                                                                                                   | 현재 파일                                     | Markdown            |

#### **필수 요구 사항 (Requirements)**

*   **주요 분석 도구:** Python (Pandas, Scikit-learn 등), STATA ( econometric estimation)
*   **문서화 도구:** Overleaf / LaTeX (for academic paper generation), JabRef (for managing citations),.

#### **인용 (Citation)**

본 연구는 서강대학교 경제학과 신혜선 교수님의 환경경제학(ECO3005) 강의를 위해 수행되었습니다.

```bibtex
@article{park2025evaluation,
  title={Evaluation of the Effectiveness of Marine Debris Reduction Policy through DID Analysis},
  author={Park, Jaehyun and Jeong, Jonghyeok},
  journal={Sogang University Environmental Economics Group Project},
  year={2025},
  note={Course: ECO3005 (2025 Fall Semester)}
}

```
## Dataset Download

**⚠️ IMPORTANT**: The dataset files (`patches/`, `predicted_unet/`, `shapefiles/`) are **not included** in this repository due to their large size.

You **must** download MARIDA from one of the following sources:

- **Primary Source**: https://doi.org/10.5281/zenodo.5151941
- **Alternative**: [Radiant MLHub](https://mlhub.earth/data/marida_v1) (includes STAC catalog)

After downloading, extract the dataset into the `data/` folder as described in [Dataset Structure](#dataset-structure).



## Installation

### Conda 환경 설치

The requirements are easily installed via
[Anaconda](https://www.anaconda.com/distribution/#download-section) (recommended):
```bash
conda env create -f environment.yml
```
> If the following error occurred: InvalidVersionSpecError: Invalid version spec: =2.7 
>
> Run: conda update conda

After the installation is completed, activate the environment:
```bash
conda activate marida
```



### Dataset Structure

**📥 Before you begin**, download [MARIDA dataset](https://doi.org/10.5281/zenodo.5151941) and extract it into the `data/` folder.

The expected directory structure after extraction:

    .
    ├── ...
    ├── data                                     # Main Dataset folder
    │   ├── patches/                             # 🔴 REQUIRED: Download from Zenodo
    │   │    ├── S2_DATE_TILE/                   # Unique Date and Tile
    │   │    │    ├── S2_DATE_TILE_CROP.tif      # 256×256 Patch (11 bands)
    │   │    │    ├── S2_DATE_TILE_CROP_cl.tif   # Classification Mask (Semantic Segmentation)
    │   │    │    └── S2_DATE_TILE_CROP_conf.tif # Annotator Confidence Level Mask
    │   │    └── ...                             # (4,143 patches total)
    │   ├── shapefiles/                          # 🔴 REQUIRED: Download from Zenodo
    │   │    └── S2_DATE_TILE.{shp,dbf,prj,...}  # Original annotation shapefiles
    │   ├── splits/                              # ✅ INCLUDED: Train/Val/Test splits
    │   │    ├── train_X.txt
    │   │    ├── val_X.txt
    │   │    └── test_X.txt
    │   ├── labels_mapping.txt                   # ✅ INCLUDED: Multi-label classification labels
    │   └── predicted_unet/                      # 📁 Empty folder (for model outputs)

## Reference
1. France 24 (2019). *Guatemala prohíbe el uso de plásticos de un solo uso.* 뉴스 기사.
2. MARIDA: [Kikaki, K. et al. (2022). *MARIDA: A benchmark for marine debris detection from Sentinel-2 remote sensing data.* PLOS ONE 17(1): e0262247.](https://doi.org/10.5194/egusphere-egu21-15243)
3. *Large-scale detection of marine debris in coastal areas with Sentinel-2*: [Rußwurm, M., Venkatesa, S. J., \& Tuia, D. *Large-scale detection of marine debris in coastal areas with Sentinel-2.* Working paper.]()
4. EnvEcon_Beamer) (GDrive): [!EnvEcon_Beamer) Evaluation of the Effectiveness of Marine Debris Reduction Policy through DID Analysis.pdf](https://drive.google.com/file/d/1J-W6gtSYffuhDKcKbGKSkkM9lHvE8buB/view?usp=sharing)

## License
This project is licensed under the MIT License.
