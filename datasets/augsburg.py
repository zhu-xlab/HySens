import torch
from torch.utils.data import DataLoader, Dataset
import numpy as np
from scipy.io import loadmat
from collections import defaultdict


def load_augsburg_wavelengths():
    hs_wavelength_augsburg = np.linspace(400, 2500, 180, dtype=np.float64)
    return hs_wavelength_augsburg


class AugsburgDataset(Dataset):

    def __init__(self, data_hs_dir, data_sar_dir, label_train_dir, label_test_dir, harmonizer, patch_size):
        self.patch_size = patch_size
        image_data_hs = loadmat(data_hs_dir)  # Load the hyperspectral image data
        image_data_sar = loadmat(data_sar_dir)  # Load SAR data
        label_data_train = loadmat(label_train_dir)  # Load the label data
        label_data_test = loadmat(label_test_dir)

        self.image_hs = image_data_hs['data_HS_LR']
        self.image_sar = image_data_sar['data_SAR_HR']
        self.label_train = label_data_train['TrainImage']
        self.label_test = label_data_test['TestImage']
        self.label = np.maximum(self.label_train, self.label_test)

        self.image_hs = harmonizer.harmonize(self.image_hs)
        self.image_sar = self.image_sar[:, :, 0:4:1]

        self.width, self.height, self.num_bands = self.image_hs.shape

        # Generate valid patch centers (i.e., where the label is not zero)
        self.valid_patches = self._get_valid_patch_centers()
        self.harmonizer = harmonizer

    def _get_valid_patch_centers(self):
        """Return list of patch center coordinates where label is not zero."""
        half_patch = self.patch_size // 2
        valid_patches = []

        for i in range(half_patch, self.width - half_patch, 1):
            for j in range(half_patch, self.height - half_patch, 1):
                if self.label[i, j] != 0:  # Only consider patches with a valid center
                    valid_patches.append((i, j))

        return valid_patches

    def __len__(self):
        # Number of valid patches
        return len(self.valid_patches)

    def __getitem__(self, idx):
        # Get the patch center coordinates
        center_x, center_y = self.valid_patches[idx]

        # Get the top-left corner of the patch
        start_x = center_x - self.patch_size // 2
        start_y = center_y - self.patch_size // 2

        # Extract the HS patch
        patch_hs = self.image_hs[start_x:start_x + self.patch_size, start_y:start_y + self.patch_size, :]

        min_vals = patch_hs.min(axis=-1, keepdims=True)  # shape: [4, 4, 1]
        max_vals = patch_hs.max(axis=-1, keepdims=True)  # shape: [4, 4, 1]
        denom = max_vals - min_vals
        denom = denom + 1e-8
        normalized_patch_hs = 2 * (patch_hs - min_vals) / denom - 1

        # Extract the corresponding label of the patch center
        label = self.label[center_x, center_y]

        # Convert pixel spectrum and label to torch tensors
        normalized_patch_hs = torch.tensor(normalized_patch_hs, dtype=torch.float32)  # Shape: [num_bands]
        label = torch.tensor(label - 1, dtype=torch.long)  # Assuming labels are class indices

        # extract the SAR patch
        patch_sar = self.image_sar[start_x:start_x + self.patch_size, start_y:start_y + self.patch_size, :]

        mean = patch_sar.mean(axis=(0, 1), keepdims=True)
        std = patch_sar.std(axis=(0, 1), keepdims=True)
        normalized_patch_sar = (patch_sar - mean) / (std + 1e-6)
        normalized_patch_sar = torch.tensor(normalized_patch_sar, dtype=torch.float32)

        return normalized_patch_hs, normalized_patch_sar, label


def create_data_loaders_augsburg(dataset, batch_size):

    # Dictionary to store indices for each class
    class_indices = defaultdict(list)

    # Populate class_indices with indices belonging to each class
    for idx in range(len(dataset)):
        _, _, label = dataset[idx]  # Get label
        class_indices[label.item()].append(idx)  # Add index to corresponding class

    # Prepare lists for train and validation indices
    train_idx = []
    val_idx = []
    test_idx = []

    # Randomly select # samples for each class for training and validation, and the rest for test
    for class_label, indices in class_indices.items():
        indices = np.array(indices)
        np.random.shuffle(indices)

        n_total = len(indices)
        n_train = 50
        n_val = 20

        train_samples = indices[:n_train]
        val_samples = indices[n_train:n_train + n_val]
        test_samples = indices[n_train + n_val:]

        train_idx.extend(train_samples)
        val_idx.extend(val_samples)
        test_idx.extend(test_samples)

    # Create train and validation datasets using Subset
    train_dataset = torch.utils.data.Subset(dataset, train_idx)
    val_dataset = torch.utils.data.Subset(dataset, val_idx)
    test_dataset = torch.utils.data.Subset(dataset, test_idx)

    # create data loader
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=True)
    return train_loader, val_loader, test_loader

