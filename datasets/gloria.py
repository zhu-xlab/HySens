import torch
from torch.utils.data import DataLoader, Dataset
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def min_max_normalize(data):
    min_data = np.min(data)
    max_data = np.max(data)
    denom = max_data - min_data
    denom = denom + 1e-8
    normalized_data = 2 * (data - min_data) / denom - 1
    return normalized_data


def load_gloria_wavelengths():
    hs_wavelength_gloria = np.linspace(350, 900, 551, dtype=np.float64)
    return hs_wavelength_gloria


class GloriaDataset(Dataset):

    def __init__(self, data_dir, label_dir, harmonizer):
        pixels = pd.read_csv(data_dir)
        labels = pd.read_csv(label_dir, usecols=['GLORIA_ID', 'Chla'])

        data_merged = pd.merge(pixels, labels, on='GLORIA_ID')

        # Filter out rows where Chla is NaN and larger than 100
        data_filtered = data_merged[data_merged['Chla'].notna()]
        data_filtered = data_filtered[~data_filtered.iloc[:, 1:-1].isna().any(axis=1)]
        data_filtered = data_filtered[data_filtered['Chla'] <= 100]

        pixels_filtered = data_filtered.iloc[:, 1:-1].values
        labels_filtered = data_filtered.iloc[:, -1].values

        pixels_filtered = harmonizer.harmonize(pixels_filtered)
        pixels_normalized = np.apply_along_axis(min_max_normalize, axis=1, arr=pixels_filtered)
        labels_filtered = labels_filtered.astype('float32')

        self.pixels = pixels_normalized
        self.labels = labels_filtered
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


def create_data_loaders_gloria(dataset, batch_size):
    all_indices = np.arange(0, len(dataset))

    cal_idx, test_idx = train_test_split(
        all_indices, test_size=0.2)

    train_idx, val_idx = train_test_split(
        cal_idx, test_size=0.25)

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