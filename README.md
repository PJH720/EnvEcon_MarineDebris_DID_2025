## 📄 Analysis of Marine Debris Reduction Policy Effectiveness (Motagua River Basin)

### **[ECO3005 Environmental Economics, Fall 2025] Final Group Project Empirical Analysis**

#### **Project Title**
*Evaluation of the Effectiveness of Marine Debris Reduction Policy through DID Analysis*

###### Presentation

- EnvEcon Beamer (GDrive):
[Evaluation of the Effectiveness of Marine Debris Reduction Policy through DID Analysis (PDF)](https://drive.google.com/file/d/1J-W6gtSYffuhDKcKbGKSkkM9lHvE8buB/view?usp=sharing)
- Korean: [한국어 README](README.ko.md)

---

#### **Course Context**
This project was conducted as a **Group Project** for the **Environmental Economics (ECO3005)** course during the Fall 2025 semester at Sogang University.

- **Course:** Environmental Economics (ECO3005)
- **Semester:** 2025 Fall
- **Instructor:** Professor Hyeseon Shin

In accordance with the **Group Project Guidelines**, this repository includes:
1) Definition of the environmental problem,
2) Data-driven economic analysis,
3) Policy evaluation and implications,
4) Comprehensive bibliography.

---

#### **Authors & Affiliation**
* **Jaehyun Park**\*, *(Department of Economics, Sogang University)*
* **Jonghyeok Jeong**†, *(Department of Economics, Sogang University)*

\* First author, † Co-author

---

#### **1. Abstract & Key Findings**

This study empirically evaluates the causal effect of the marine plastic blocking and collection policy (implemented on October 31, 2019) in the **Motagua River Basin**, Guatemala. We utilized the **Marine Debris Ratio** as a quantitative indicator, derived from a deep learning model trained on the European Space Agency (ESA) **Sentinel-2** satellite imagery and the **MARIDA** dataset.

The results of the **Difference-in-Differences (DID)** analysis are as follows:
- $\beta_3 \approx -0.095$ (Direction of reduction effect),
- p-value ≈ 0.64 (**Statistically Insignificant**)

The policy's effect was **not statistically significant**. Following the policy, the debris ratio in Motagua showed a slight upward trend in the time-series pattern, making it difficult to confirm a clear reduction effect within the observation period. Notably, a critical structural limitation exists: the control group sample size was extremely small after the policy implementation, affecting the power and assumptions of the DID model.

| Category | Description |
| :--- | :--- |
| **Core Research Question** | Did the Motagua regional policy significantly reduce the marine debris ratio? |
| **Analytical Methods** | Sentinel-2 + MARIDA U-Net, Difference-in-Differences (DID), Seasonal adjustment, Correlation analysis |
| **Policy Effect (DID Estimate)** | **-0.095363** (Indicates a negative direction/reduction, but **Insignificant** with p-value > 0.6) |
| **Policy Interpretation** | Difficult to conclude a definitive policy effect within the observed period due to data and identification limitations. |

---

#### **2. Environmental Economic Framework**

This research justifies the marine pollution problem based on core concepts of environmental economics:

1.  **Negative Externality & Market Failure**
    * Marine plastic pollution is a **negative externality** where producers and consumers do not account for the social damage costs (Marginal Damage, MD).
    * This causes the **Social Marginal Cost (SMC)** to exceed the **Private Marginal Cost (PMC)**, leading to a **market failure** where the market generates excessive pollution beyond the **socially optimal level ($Q^*$)**, resulting in **Deadweight Loss (DWL)**.
2.  **Tragedy of the Commons**
    * Marine and river ecosystems are **Open-access Resources** characterized by non-excludability and rivalry.
    * High transaction costs prevent private negotiation, leading to the **Tragedy of the Commons** where individual self-interest results in over-exploitation and resource depletion, providing a strong justification for government intervention.
3.  **Non-market Valuation & Cost-Benefit Analysis**
    * Plastic pollution affects the value of non-market goods such as tourism, fisheries, and residential environments, making it a classic application of Cost-Benefit Analysis (CBA) and non-market valuation techniques.

---

#### **3. Data and Technical Approach**

This study leveraged **Python** programming and **Big Data processing** to construct remote sensing data into a format suitable for economic analysis.

1.  **Dependent Variable Construction (Remote Sensing)**
    * **Data Source:** **Sentinel-2** multi-spectral satellite imagery from the European Space Agency (ESA).
    * **Training Data:** The **MARIDA (Marine Debris Archive)** benchmark dataset was used. MARIDA is sophisticatedly annotated to distinguish plastic debris from 15 other marine features (algae, ships, foam, etc.).
    * **Model:** A **U-Net** based Semantic Segmentation model was utilized to estimate the debris ratio at the pixel level.
2.  **DID Analysis Setup**
    * **Analysis Period:** 2016‑07‑17 ~ 2020‑12‑22 (N=359)
    * **Treatment Group:** Motagua River Basin (Tile 16PCC).
    * **Control Group:** All other regions.
    * **Policy Date:** October 31, 2019 (Guatemala single-use plastic regulation & plastic barrier installation).
    * **DID Model:**

$$
Y_{it} = \beta_0 + \beta_1 \text{Treated}_i + \beta_2 \text{Post}_t + \beta_3 (\text{Treated}_i \times \text{Post}_t) + \epsilon_{it}
$$

* $Y_{it}$ represents the debris ratio for region $i$ at time $t$.
* Time-series correlation and **Seasonal Adjustment** were applied to monitor dynamics before and after the policy.

---

#### **4. Empirical Results & Limitations**

| Item | Treatment (Motagua) | Control (Other Regions) |
| :--- | :--- | :--- |
| **Pre-policy Mean** | 0.161439 (n=121) | 0.105041 (n=168) |
| **Post-policy Mean** | 0.183842 (n=68) | **0.222807 (n=2)** |
| **Change** | +0.022403 (Increase) | +0.117766 (Increase) |
| **DID ($\beta_3$)** | **-0.095363** (p=0.6361) | |

1.  **Inconclusive Policy Effect**
    * While the DID estimate was -0.0954 (suggesting a decreasing trend), the p-value of 0.6361 **failed to meet the statistical significance threshold ($p < 0.05$)**. We cannot reject the **null hypothesis** that the policy had no effect.
2.  **Critical Data & Methodological Limitations**
    * **Severe Lack of Control Group Samples:** The number of samples for the control group post-policy ($n=2$) was extremely low. Consequently, the mean change for the control group was calculated under **extreme volatility**, severely undermining the precision and credibility of the DID estimate ($\beta_3$).
    * **Dynamic Trends:** The debris ratio in the Motagua region showed a significant increase over time following policy implementation ($r=0.2410, p=0.0477$). This suggests potential dynamic trends that require further in-depth analysis beyond the standard DID.

---

#### **5. Policy Implications & Future Work**

- **Econometric Reinforcement (Causal Inference)**
    1.  **Enhancing Power:** To validate the current effect size, the effective sample size needs to increase by approximately **17 times**. Future research should incorporate long-term panel data or address the sample imbalance.
    2.  **Dynamic Analysis:** Apply an **Event Study** design including **Fixed Effects** (region and time) to test the parallel trends assumption.
- **Future Directions:**
    1.  **Data Expansion:** Include longer timeframes and more tiles/observations to ensure sample size and control group diversity.
    2.  **Economic Indicator Integration:** Design analyses linked to real economic variables like tourism revenue, fishery yields, and housing prices.

---

#### **6. Project Structure & Reproducibility**

This repository contains all code, data summaries, results, and academic materials used in the analysis.

| Directory/File | Description | Tools Used |
| :--- | :--- | :--- |
| `!EnvEcon_Beamer) Evaluation of the Effectiveness... .pdf` | **Final Research Report (Original PDF)** | Overleaf / BibTeX |
| `did_analysis_motagua/` | Visualizations and statistical summaries of Motagua DID | Python / Matplotlib |
| `regional_distribution.png` | Distribution of debris ratio and samples by region | Python |
| `temporal_distribution.png` | Time-series analysis (Annual/Seasonal) of debris ratio | Python |
| `analysis_summary.txt` / `cost_benefit_summary.md` | Summary of DID and hypothetical Cost-Benefit Analysis | Python |
| `README.md` | Current file | Markdown |

#### **Requirements**
* **Main Analysis:** Python (Pandas, Scikit-learn, etc.), STATA (for econometric estimation)
* **Documentation:** Overleaf / LaTeX, JabRef (Citation management)

#### **Citation**
This research was conducted for the Environmental Economics (ECO3005) course taught by Professor Hyeseon Shin at Sogang University.

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

### Conda Environment Setup

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
1. France 24 (2019). *Guatemala prohíbe el uso de plásticos de un solo uso.* [뉴스 기사](https://www.france24.com/es/20190920-guatemala-prohibe-plastico-un-solo-uso).
2. MARIDA: [Kikaki, K. et al. (2022). *MARIDA: A benchmark for marine debris detection from Sentinel-2 remote sensing data.* PLOS ONE 17(1): e0262247.](https://doi.org/10.5194/egusphere-egu21-15243)
3. *Large-scale detection of marine debris in coastal areas with Sentinel-2*: [Rußwurm, M., Venkatesa, S. J., \& Tuia, D. *Large-scale detection of marine debris in coastal areas with Sentinel-2.* Working paper.](https://arxiv.org/html/2307.02465)
4. EnvEcon_Beamer) (GDrive): [!EnvEcon_Beamer) Evaluation of the Effectiveness of Marine Debris Reduction Policy through DID Analysis.pdf](https://drive.google.com/file/d/1J-W6gtSYffuhDKcKbGKSkkM9lHvE8buB/view?usp=sharing)

## License
This project is licensed under the MIT License.
