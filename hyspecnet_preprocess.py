# read the Hyspecnet data, preprocess all patches and convert into numpy
import numpy as np
import rasterio
import os


# read the hyperspectral patch tif file
def read_patch(data_dir, output_dir):
    index = 1
    # Loop through each folder and subfolder to find the .TIF files
    for foldername, subfolders, filenames in os.walk(data_dir):
        for subfolder in subfolders:
            subfolder_path = os.path.join(foldername, subfolder)

            # Go through files in the subfolder
            for file in os.listdir(subfolder_path):
                if file.endswith("SPECTRAL_IMAGE.TIF"):
                    tif_path = os.path.join(subfolder_path, file)
                    invalid_channels = [126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 160,
                                        161, 162, 163, 164, 165, 166]
                    valid_channels_ids = [c + 1 for c in range(224) if c not in invalid_channels]

                    minimum_value = 0
                    maximum_value = 10000
                    # Open the SPECTRAL_IMAGE.TIF using rasterio and convert to numpy
                    with rasterio.open(tif_path) as src:
                        img_array = src.read(valid_channels_ids)
                        # clip data to remove uncertainties
                        clipped = np.clip(img_array, a_min=minimum_value, a_max=maximum_value)
                        # min-max normalization
                        min_vals = clipped.min(axis=0, keepdims=True)
                        max_vals = clipped.max(axis=0, keepdims=True)

                        # Avoid division by zero by ensuring the denominator is non-zero
                        denom = np.where(max_vals - min_vals == 0, 1, max_vals - min_vals)

                        # Apply the min-max normalization: (x - min) / (max - min)
                        out_data = 2 * (clipped - min_vals) / denom - 1

                        out_data = out_data.astype(np.float32)
                        print(f"Processed file: {tif_path}")

                        # Save the numpy array
                        np_save_path = os.path.join(output_dir, f"image_{index}.npy")
                        # Save the numpy array
                        np.save(np_save_path, out_data)

                        index += 1


# main
DATA_DIR = './data_pretrain/hyspecnet'
OUTPUT_DIR = './data_pretrain/preprocess_patches'
read_patch(DATA_DIR, OUTPUT_DIR)