import torch
from torch.utils.data import DataLoader, Dataset
import numpy as np
from scipy.io import loadmat
from collections import defaultdict


def load_bay_wavelengths():
    bay_wavelength = np.linspace(400, 2500, 224, dtype=np.float32)
    return bay_wavelength


class BayDataset(Dataset):

    def __init__(self, data_dir_pre, data_dir_post, label_dir, harmonizer, patch_size):
        self.patch_size = patch_size
        # Load the hyperspectral image and labels from .mat files
        image_data_pre = loadmat(data_dir_pre)  # Load the hyperspectral image data
        image_data_post = loadmat(data_dir_post)
        label_data = loadmat(label_dir)  # Load the label data

        self.image_pre = image_data_pre['HypeRvieW'][:, :, :]
        self.image_post = image_data_post['HypeRvieW'][:, :, :]
        self.label = label_data['HypeRvieW']

        self.image_pre = harmonizer.harmonize(self.image_pre)
        self.image_post = harmonizer.harmonize(self.image_post)

        self.width, self.height, self.num_bands = self.image_pre.shape

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
        patch_pre = self.image_pre[start_x:start_x + self.patch_size, start_y:start_y + self.patch_size, :]
        patch_post = self.image_post[start_x:start_x + self.patch_size, start_y:start_y + self.patch_size, :]

        min_vals_pre = patch_pre.min(axis=-1, keepdims=True)
        max_vals_pre = patch_pre.max(axis=-1, keepdims=True)
        denom_pre = max_vals_pre - min_vals_pre
        denom_pre = denom_pre + 1e-8
        normalized_patch_pre = 2 * (patch_pre - min_vals_pre) / denom_pre - 1

        min_vals_post = patch_post.min(axis=-1, keepdims=True)
        max_vals_post = patch_post.max(axis=-1, keepdims=True)
        denom_post = max_vals_post - min_vals_post
        denom_post = denom_post + 1e-8
        normalized_patch_post = 2 * (patch_post - min_vals_post) / denom_post - 1

        # Extract the corresponding label of the patch center
        label = self.label[center_x, center_y]

        # Convert pixel spectrum and label to torch tensors
        normalized_patch_pre = torch.tensor(normalized_patch_pre, dtype=torch.float32)
        normalized_patch_post = torch.tensor(normalized_patch_post, dtype=torch.float32)
        label = torch.tensor(label - 1, dtype=torch.long)  # Assuming labels are class indices

        return normalized_patch_pre, normalized_patch_post, label


def create_data_loaders_bay(dataset, batch_size):
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
        n_train = 100
        n_val = 50

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