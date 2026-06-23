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
| Classification | WHU-Hi-HóngHu | https://rsidea.whu.edu.cn/resource_WHUHi_sharing.htm |
| Change Detection | Hermiston | https://gitlab.citius.gal/HPC4RS/ChangeDetectionDataset/ |
| Change Detection | Bay Area | https://gitlab.citius.gal/HPC4RS/ChangeDetectionDataset/ |
| Change Detection | Santa Barbara | https://gitlab.citius.gal/HPC4RS/ChangeDetectionDataset/ |
| Modalities Fusion | Augsburg | https://github.com/danfenghong/ISPRS_S2FL |
| Modalities Fusion | Berlin | https://github.com/danfenghong/ISPRS_S2FL |
| Regression | GLORIA | ... |
| Regression | KarLy | ... |


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

