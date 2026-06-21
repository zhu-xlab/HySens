import torch
from torch.utils.data import DataLoader, Dataset
import numpy as np
from scipy.io import loadmat
from collections import defaultdict


def load_hermiston_wavelengths():
    hermiston_wavelengths = np.array([
        426.82000, 435.93427, 445.04852, 454.16278, 463.27704, 472.39130,
        481.50555, 490.61980, 499.73407, 508.84833, 517.96260, 527.07684,
        536.19110, 545.30536, 554.41960, 563.53390, 572.64813, 581.76240,
        590.87665, 599.99090, 609.10516, 618.21940, 627.33370, 636.44794,
        645.56220, 654.67645, 663.79070, 672.90500, 682.01930, 691.13354,
        700.24780, 709.36206, 718.47630, 727.59060, 736.70483, 745.81910,
        754.93335, 764.04760, 773.16187, 782.27610, 791.39040, 800.50464,
        809.61890, 818.73315, 827.84740, 836.96170, 846.07590, 855.19020,
        864.30444, 873.41870, 882.53296, 891.64720, 900.76150, 909.87573,
        918.99000, 928.10425, 937.21850, 946.33276, 955.44700, 964.56130,
        973.67554, 982.78980, 991.90405, 1001.01830, 1010.13257, 1019.24680,
        1028.36110, 1037.47530, 1046.58960, 1055.70390, 1064.81810, 1073.93240,
        1083.04660, 1092.16090, 1101.27510, 1110.38940, 1119.50370, 1128.61790,
        1137.73220, 1146.84640, 1155.96070, 1165.07500, 1174.18920, 1183.30350,
        1192.41770, 1201.53200, 1210.64620, 1219.76050, 1228.87480, 1237.98900,
        1247.10340, 1256.21770, 1265.33190, 1274.44620, 1283.56040, 1292.67470,
        1301.78900, 1310.90320, 1320.01750, 1329.13170, 1338.24600, 1347.36020,
        1356.47450, 1365.58870, 1374.70300, 1383.81730, 1392.93150, 1402.04580,
        1411.16000, 1420.27430, 1429.38850, 1438.50280, 1447.61710, 1456.73130,
        1465.84560, 1474.95980, 1484.07410, 1493.18840, 1502.30260, 1511.41690,
        1520.53110, 1529.64540, 1538.75960, 1547.87390, 1556.98820, 1566.10240,
        1575.21670, 1584.33090, 1593.44520, 1602.55940, 1611.67370, 1620.78800,
        1629.90220, 1639.01650, 1648.13070, 1657.24500, 1666.35930, 1675.47350,
        1684.58780, 1693.70200, 1702.81630, 1711.93050, 1721.04480, 1730.15900,
        1739.27330, 1748.38760, 1757.50180, 1766.61610, 1775.73030, 1784.84460,
        1793.95890, 1803.07310, 1812.18740, 1821.30160, 1830.41590, 1839.53020,
        1848.64440, 1857.75870, 1866.87290, 1875.98720, 1885.10140, 1894.21570,
        1903.33000, 1912.44420, 1921.55850, 1930.67270, 1939.78700, 1948.90120,
        1958.01550, 1967.12980, 1976.24400, 1985.35830, 1994.47250, 2003.58680,
        2012.70120, 2021.81540, 2030.92970, 2040.04400, 2049.15820, 2058.27250,
        2067.38670, 2076.50100, 2085.61520, 2094.72950, 2103.84380, 2112.95800,
        2122.07230, 2131.18650, 2140.30080, 2149.41500, 2158.52930, 2167.64360,
        2176.75780, 2185.87200, 2194.98630, 2204.10060, 2213.21480, 2222.32900,
        2231.44340, 2240.55760, 2249.67190, 2258.78610, 2267.90040, 2277.01460,
        2286.12900, 2295.24320, 2304.35740, 2313.47170, 2322.58600, 2331.70020,
        2340.81450, 2349.92870, 2359.04300, 2368.15720, 2377.27150, 2386.38570,
        2395.50000
    ], dtype=np.float64)
    return hermiston_wavelengths


class HermistonDataset(Dataset):

    def __init__(self, data_dir_pre, data_dir_post, label_dir, harmonizer, patch_size):
        self.patch_size = patch_size
        # Load the hyperspectral image and labels from .mat files
        image_data_pre = loadmat(data_dir_pre)  # Load the hyperspectral image data
        image_data_post = loadmat(data_dir_post)
        label_data = loadmat(label_dir)  # Load the label data

        self.image_pre = image_data_pre['HypeRvieW'][:, :, :]
        self.image_post = image_data_post['HypeRvieW'][:, :, :]
        self.label = label_data['gt5clasesHermiston']

        self.image_pre = harmonizer.harmonize(self.image_pre)
        self.image_post = harmonizer.harmonize(self.image_post)

        # binarize the labels
        binary_labels = self.label.copy()
        binary_labels[np.isin(self.label, [2, 3, 4, 5])] = 2
        self.label = binary_labels
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


def create_data_loaders_hermiston(dataset, batch_size):
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