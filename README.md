![Marine Debris Archive Logo](./docs/marida_trans.png)

# MARIDA: Marine Debris Archive

Marine Debris Archive (MARIDA) is a comprehensive marine debris-oriented benchmark dataset on Sentinel-2 satellite images. It also includes various sea features that co-exist with marine debris in coastal and offshore environments.

## Dataset Overview

- **Temporal Coverage**: 2015-2021
- **Spatial Coverage**: 17 Sentinel-2 tiles across 12 countries, globally distributed marine pollution hotspots
- **Format**: GeoTIFF (.tif) files with 11 spectral bands
- **Resolution**: 256×256 pixel patches
- **Classes**: 15 distinct marine and atmospheric classes (Marine Debris, Sargassum, Ships, Water types, etc.)
- **Task Focus**: Weakly supervised pixel-level semantic segmentation

This repository hosts the basic tools for the extraction of spectral signatures as well as the code for the reproduction of the baseline models (U-Net, Random Forest, ResNet).

## Citation

If you find this repository useful, please consider giving a star :star: and citation:
 > Kikaki K, Kakogeorgiou I, Mikeli P, Raitsos DE, Karantzalos K (2022) MARIDA: A benchmark for Marine Debris detection from Sentinel-2 remote sensing data. PLoS ONE 17(1): e0262247. https://doi.org/10.1371/journal.pone.0262247

## Dataset Download

**⚠️ IMPORTANT**: The dataset files (`patches/`, `predicted_unet/`, `shapefiles/`) are **not included** in this repository due to their large size.

You **must** download MARIDA from one of the following sources:

- **Primary Source**: https://doi.org/10.5281/zenodo.5151941
- **Alternative**: [Radiant MLHub](https://mlhub.earth/data/marida_v1) (includes STAC catalog)

After downloading, extract the dataset into the `data/` folder as described in [Dataset Structure](#dataset-structure).


## Contents

- [Installation](#installation)
	- [Installation Requirements](#installation-requirements)
	- [Installation Guide](#installation-guide)
- [Getting Started](#getting-started)
	- [Dataset Structure](#dataset-structure)
	- [Spectral Signatures Extraction](#spectral-signatures-extraction)
	- [Weakly Supervised Pixel-Level Semantic Segmentation](#weakly-supervised-pixel-Level-semantic-segmentation)
		- [Unet](#unet)
		- [Random Forest](#random-forest)
	- [Multi-label Classification](#multi-label-classification)
		- [ResNet](#resnet)
- [MARIDA - Exploratory Analysis](https://marine-debris.github.io/)
- [Talks and Papers](#talks-and-papers)


## Installation

### Installation Requirements
- python == 3.7.10
- pytorch == 1.7 
- cudatoolkit == 11.0 (For GPU usage, compute capability >= 3.5)
- gdal == 2.3.3
- rasterio == 1.0.21
- scikit-learn == 0.24.2
- numpy == 1.20.2
- tensorboard == 1.15
- torchvision == 0.8.0
- scikit-image == 0.18.1
- pandas == 1.2.4
- pytables == 3.6.1
- tqdm == 4.59.0


### Installation Guide

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

## Getting Started

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

> **Note**: `patches/` and `shapefiles/` folders contain large files (~several GB) and must be downloaded separately from [Zenodo](https://doi.org/10.5281/zenodo.5151941).


The mapping in S2_DATA_TILE_CROP_cl between Digital Numbers and Classes is:

```yaml
1: 'Marine Debris',
2: 'Dense Sargassum',
3: 'Sparse Sargassum',
4: 'Natural Organic Material',
5: 'Ship',
6: 'Clouds',
7: 'Marine Water',
8: 'Sediment-Laden Water',
9: 'Foam',
10: 'Turbid Water',
11: 'Shallow Water',
12: 'Waves',
13: 'Cloud Shadows',
14: 'Wakes',
15: 'Mixed Water'
```

For the confidence level mask or other usefull mappings go to utils/assets.py

Also, in order to easily visualize the RGB composite of the S2_DATE_TILE_CROP patches via [QGIS](https://qgis.org/en/site/index.html),
you can use the `utils/qgis_color_patch_rgb.qml` file.

### Spectral Signatures Extraction

For the extraction of the spectal signature of each annotated pixel and
its storage in a HDF5 Table file (DataFrame-like processing) run the following commands below. 
The output `data/dataset.h5` can be used for the spectral analysis of the dataset.
Also, this stage is required for the Random Forest training (press [here](#random-forest)). 
Note that this is not required for the Unet training. This procedure lasts approximately ~10 minutes.

```bash
python utils/spectral_extraction.py
```

Alternatively, you can download the `dataset.h5` file from [here](https://pithos.okeanos.grnet.gr/public/sbT8ASX0HINAdx4tmKCg27) and put it in the `data` folder.
Finally, in order to load the `dataset.h5` with Pandas, run in a python cell the following:

```python
import pandas as pd

hdf = pd.HDFStore('./data/dataset.h5', mode = 'r')

df_train = hdf.select('train')
df_val = hdf.select('val')
df_test = hdf.select('test')

hdf.close()
```

### Weakly Supervised Pixel-Level Semantic Segmentation

#### Unet

**Unet training**

Spectral Signatures Extraction in not required for this procedure.
For training in the "train" set and evaluation in "val" set with the proposed parameters, run:

```bash
cd semantic_segmentation/unet
python train.py
```

While training, in order to see the loss status and various metrics via tensorboard, run in a different terminal 
the following command and then go to `localhost:6006` with your browser:

```bash
tensorboard --logdir logs/tsboard_segm
```

The `train.py` also supports the following argument flags:

```bash
    # Basic parameters
    --agg_to_water "Aggregate Mixed Water, Wakes, Cloud Shadows, Waves with Marine Water (True or False)"
    --mode "Select between 'train' or 'test'"
    --epochs "Number of epochs to run"
    --batch "Batch size"
    --resume_from_epoch "Load model from previous epoch (To continue the training)"
    
    # Unet
    --input_channels "The number of input bands"
    --output_channels "The number of output classes"
    --hidden_channels "The number of hidden features"

    # Optimization
    --weight_param "Weighting parameter for Loss Function"
    --lr "Learning rate for adam"
    --decay "Learning rate decay for adam"
    --reduce_lr_on_plateau "Reduce learning rate when val loss no decrease (0 or 1)"
    --lr_steps "Specify the steps that the lr will be reduced"

    # Evaluation/Checkpointing
    --checkpoint_path "The folder to save checkpoints into."
    --eval_every "How frequently to run evaluation (epochs)"

    # misc
    --num_workers "How many cpus for loading data (0 is the main process)"
    --pin_memory "Use pinned memory or not"
    --prefetch_factor "Number of sample loaded in advance by each worker"
    --persistent_workers "This allows to maintain the workers Dataset instances alive"
    --tensorboard "Name for tensorboard run"
```

**Unet evaluation**

Run the following commands in order to produce the Confusion Matrix in stdout and `logs/evaluation_unet.log`,
 as well as to produce the predicted masks from the test set in `data/predicted_unet/` folder:

```bash
cd semantic_segmentation/unet
python evaluation.py
```

In order to easily visualize the predicted masks via [QGIS](https://qgis.org/en/site/index.html),
you can use the `utils/qgis_color_mask_mapping.qml` file.

To download the pretrained Unet model on MARIDA press [here](https://pithos.okeanos.grnet.gr/public/lxh8hL4zvuSKds2BdVnMd2). 
Then, you should put these items in the `semantic_segmentation/unet/trained_models/` folder.

#### Random Forest

In our baseline setup we trained a random forest classifier on Spectral Signatures,
produced Spectral Indices (SI) and extracted Gray-Level Co-occurrence Matrix (GLCM) texture features.
Thus, this process requires the Spectral Signatures Extraction i.e., the `data/dataset.h5` [file](#spectral-signatures-extraction). Also, it requires the `dataset_si.h5` and `dataset_glcm.h5` for SI and GLCM features,
respectively.

1) For the extraction of stacked SI patches (in `data/indices/`) run:

```bash
cd semantic_segmentation/random_forest
python engineering_patches.py
```

Then, in order to produce the `dataset_si.h5` run:

```bash
python utils/spectral_extraction.py --type indices
```

2) For the stacked GLCM patches (in `data/texture/`) run (approximately ~ 110 mins):

```bash
python engineering_patches.py --type texture
```

Similarly, in order to produce the `dataset_glcm.h5` run:

```bash
python utils/spectral_extraction.py --type texture
```

 Alternatively, you can download the `indices/` and `texture/` folders as well as the `dataset_si.h5` and `dataset_glcm.h5` files from [here](https://pithos.okeanos.grnet.gr/public/7Xm6x2uSBHTknNv7vaqgS6). 
Then, you should put these items in the `data` folder.

**Random Forest training and evaluation**

For training in "train" set and final evaluation in "test" set, run the following commands.
Note that the results will appear in stdout and `logs/evaluation_rf.log`, and the predicted 
masks in `data/predicted_rf/` folder.

```bash
cd semantic_segmentation\random_forest
python train_eval.py
```

The `train_eval.py` supports the `--agg_to_water` argument for 
the aggregation of various classes to form the Water Super Class (The default setup):

```bash
python train_eval.py --agg_to_water ['"Mixed Water"','"Wakes"','"Cloud Shadows"','"Waves"']
```

### Multi-label Classification

The weakly-supervised multi-label classification task is an incomplete multi-label
assignment problem. Specifically, the assigned labels are definitely positive (assigned as 1),
 while the absent labels (assigned as 0) are not necessarily negative. The assigned labels
 per patch can be found in `data/labels_mapping.txt`

#### ResNet

**ResNet training**

For training in "train" set and evaluation in "val" set, run:

```bash
cd multi-label/resnet
python train.py
```

Similarly to U-Net training, you can use tensorboard thought `localhost:6006` 
to visualize the training process:

```
tensorboard --logdir logs/tsboard_multilabel
```

**ResNet evaluation**

Run the following commands in order to produce the accuracy scores and the Confusion Matrix in stdout 
and `logs/evaluation_resnet.log`, as well as to produce the predictions for each patch from the test 
set in `data/predicted_labels_mapping.txt`:

```bash
python evaluation.py
```

To download the pretrained ResNet model on MARIDA press [here](https://pithos.okeanos.grnet.gr/public/lxh8hL4zvuSKds2BdVnMd2). 
Then, you should put these items in the `multi-label/resnet/trained_models/` folder.

## Presentations
[Kikaki A, Kakogeorgiou I, Mikeli P, Raitsos DE, Karantzalos K. Detecting and Classifying Marine Plastic Debris from high-resolution multispectral satellite data.](https://doi.org/10.5194/egusphere-egu21-15243)

## License
This project is licensed under the MIT License.
