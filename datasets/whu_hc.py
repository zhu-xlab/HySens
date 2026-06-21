import torch
from torch.utils.data import DataLoader, Dataset
import numpy as np
import tifffile as tiff
from collections import defaultdict


def load_whuhc_wavelengths():
    hs_wavelength_whu = np.linspace(400, 1000, 274, dtype=np.float64)
    return hs_wavelength_whu


class WHUHC_Dataset(Dataset):

    def __init__(self, data_dir, label_dir, harmonizer, patch_size):
        self.patch_size = patch_size
        self.image = tiff.imread(data_dir)
        self.label = tiff.imread(label_dir)

        # Transpose the image to [width, height, bands]
        self.image = self.image.transpose(1, 2, 0)
        self.image = harmonizer.harmonize(self.image)

        # Only use regions where label is not zero
        self.label = self.label
        self.width, self.height, self.num_bands = self.image.shape

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

        # Extract the patch
        patch = self.image[start_x:start_x + self.patch_size, start_y:start_y + self.patch_size, :]

        min_vals = patch.min(axis=-1, keepdims=True)
        max_vals = patch.max(axis=-1, keepdims=True)
        denom = max_vals - min_vals
        denom = denom + 1e-8
        normalized_patch = 2 * (patch - min_vals) / denom - 1

        # Extract the corresponding label of the patch center
        label = self.label[center_x, center_y]

        # Convert pixel spectrum and label to torch tensors
        normalized_patch = torch.tensor(normalized_patch, dtype=torch.float32)
        label = torch.tensor(label - 1, dtype=torch.long)  # Assuming labels are class indices

        return normalized_patch, label


def create_data_loaders_whuhc(dataset, batch_size):
    # Dictionary to store indices for each class
    class_indices = defaultdict(list)

    # Populate class_indices with indices belonging to each class
    for idx in range(len(dataset)):
        _, label = dataset[idx]  # Get label
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
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True,
                              drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=True)
    return train_loader, val_loader, test_loader