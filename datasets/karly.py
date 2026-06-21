import torch
from torch.utils.data import DataLoader, Dataset
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def load_karly_wavelengths():
    # read soil moisture dataset
    hs_wavelengths_karly = np.array([
        454, 458, 462, 466, 470, 474, 478, 482, 486, 490,
        494, 498, 502, 506, 510, 514, 518, 522, 526, 530,
        534, 538, 542, 546, 550, 554, 558, 562, 566, 570,
        574, 578, 582, 586, 590, 594, 598, 602, 606, 610,
        614, 618, 622, 626, 630, 634, 638, 642, 646, 650,
        654, 658, 662, 666, 670, 674, 678, 682, 686, 690,
        694, 698, 702, 706, 710, 714, 718, 722, 726, 730,
        734, 738, 742, 746, 750, 754, 758, 762, 766, 770,
        774, 778, 782, 786, 790, 794, 798, 802, 806, 810,
        814, 818, 822, 826, 830, 834, 838, 842, 846, 850,
        854, 858, 862, 866, 870, 874, 878, 882, 886, 890,
        894, 898, 902, 906, 910, 914, 918, 922, 926, 930,
        934, 938, 942, 946, 950
    ])
    return hs_wavelengths_karly[1:121:1]


class KarlyDataset(Dataset):

    def __init__(self, data_dir, label_dir, harmonizer):

        df = pd.read_csv(data_dir, index_col=0)
        # get hyperspectral bands:
        hypbands = []
        for col in df.columns:
            try:
                int(col)
            except Exception:
                continue
            hypbands.append(col)

        pixels = df[hypbands]
        labels = df["soil_moisture"]

        pixels = np.asarray(pixels)
        pixels = pixels[:, :]
        labels = np.asarray(labels, dtype=np.float32)

        pixels_min = pixels.min(axis=1, keepdims=True)
        pixels_max = pixels.max(axis=1, keepdims=True)
        pixels_normalized = 2 * (pixels - pixels_min) / (pixels_max - pixels_min + 1e-8) - 1
        pixels_normalized = harmonizer.harmonize(pixels_normalized)

        self.pixels = pixels_normalized
        self.labels = labels
        self.sample_num, self.num_bands = pixels_normalized.shape
        self.harmonizer = harmonizer

    def __len__(self):
        # Number of valid patches
        return self.sample_num

    def __getitem__(self, idx):
        # Extract the patch
        pixel = self.pixels[idx, :]
        # Extract the corresponding label of the patch center
        label = self.labels[idx]
        # Convert pixel spectrum and label to torch tensors
        pixel = torch.tensor(pixel, dtype=torch.float32)  # Shape: [num_bands]
        return pixel, label


def create_data_loaders_karly(dataset, batch_size):
    all_indices = np.arange(len(dataset))

    cal_idx, test_idx = train_test_split(
        all_indices, test_size=0.2, random_state=42)

    train_idx, val_idx = train_test_split(
        cal_idx, test_size=0.25, random_state=42)

    # Create train and validation datasets using Subset
    train_dataset = torch.utils.data.Subset(dataset, train_idx)
    val_dataset = torch.utils.data.Subset(dataset, val_idx)
    test_dataset = torch.utils.data.Subset(dataset, test_idx)

    # create data loader
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True,
                              drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=True)
    return train_loader, val_loader, test_loader