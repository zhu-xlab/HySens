# HySens
PyTorch implementation of "HySens: Sensor-Agnostic Foundation Models for Hyperspectral Data (IEEE TGRS)"

## Dataset Preparation

### Pretraining Datasets

The pretraining datasets (HySpecNet-11k) used in this work can be downloaded from the following sources:
https://hyspecnet.rsim.berlin/

After downloading, organize the files as follows:

```text
data_pretrain/
└── hyspecnet/
    ├── ENMAP_Scene_1/
    ├── ENMAP_Scene_2/
    ├── ENMAP_Scene_3/
    └── ...
```
### Downstream Datasets

The following benchmark datasets are supported:

| Task | Dataset | Download |
|--------|---------|----------|
| Classification | Pavia University | https://www.ehu.eus/ccwintco/index.php/Hyperspectral_Remote_Sensing_Scenes |
| Classification | Indian Pines | https://www.ehu.eus/ccwintco/index.php/Hyperspectral_Remote_Sensing_Scenes |
| Classification | WHU-Hi-HanChuan | https://rsidea.whu.edu.cn/resource_WHUHi_sharing.htm |
| Classification | WHU-Hi-HongHu | https://rsidea.whu.edu.cn/resource_WHUHi_sharing.htm |
| Change Detection | Hermiston | https://gitlab.citius.gal/HPC4RS/ChangeDetectionDataset/ |
| Change Detection | Bay Area | https://gitlab.citius.gal/HPC4RS/ChangeDetectionDataset/ |
| Change Detection | Santa Barbara | https://gitlab.citius.gal/HPC4RS/ChangeDetectionDataset/ |
| Modalities Fusion | Augsburg | https://github.com/danfenghong/ISPRS_S2FL |
| Modalities Fusion | Berlin | https://github.com/danfenghong/ISPRS_S2FL |
| Regression | GLORIA | https://doi.pangaea.de/10.1594/PANGAEA.948492 |
| Regression | KarLy | https://github.com/felixriese/hyperspectral-soilmoisture-dataset |

After downloading, organize the files as follows:

```text
data_downstream/
├── Pavia/
│   ├── PaviaU.mat
│   └── PaviaU_gt.mat
│
├── BayArea/
│   ├── Bay_Area_2013.mat
│   └── Bay_Area_2015.mat
│   └── bayArea_gtChanges2.mat
│
├── HS-SAR-DSM Augsburg/
│   ├── data_HS_LR.mat
│   └── data_SAR_HR.mat
│   └── TrainImage.mat
│   └── TestImage.mat
│
├── Gloria_2022/
│   ├── GLORIA_Rrs.csv
│   └── GLORIA_meta_and_lab.csv
│
└── ...
```

## Pretraining

Run self-supervised pretraining:

```bash
python hyspecnet_preprocess.py
python pretrain.py
```

## Finetuning

### Classification
```bash
python finetune_classification.py --dataset pavia --tuning mode full_finetuning
```

### Change Detection
```bash
python finetune_change_detection.py --dataset hermiston --tuning mode freeze_embedding
```

### Modality Fusion
```bash
python finetune_fusion.py --dataset augsburg --tuning mode scratch
```

### Regression
```bash
python finetune_regression.py --dataset gloria --tuning mode linear_probing
```

## Citation
If you find this work useful in your research, please cite:
```
@ARTICLE{11514078,
  author={Zhao, Xiangyu and Xiong, Zhitong and Zhu, Xiao Xiang},
  journal={IEEE Transactions on Geoscience and Remote Sensing}, 
  title={HySens: Sensor-Agnostic Foundation Models for Hyperspectral Data}, 
  year={2026},
  volume={64},
  number={},
  pages={5513915-5513915},
  keywords={Modeling;Frequency modulation;Image sensors;Remote sensing;Transformers;Design methodology;Pixel;Hyperspectral imaging;Distance measurement;Hyperspectral sensors;Foundation model (FM);hyperspectral;sensor agnostic;spectroscopy},
  doi={10.1109/TGRS.2026.3691782}}

