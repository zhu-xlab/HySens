import numpy as np
from scipy.interpolate import interp1d


# spectral harmonizer for pretraining
# create full wavelength basis from 400 to 2500
class SpectralHarmonizer:
    """
    Spectral Harmonizer
    Spectral interpolation to target wavelength grid
    """

    def __init__(self, source_wavelengths, target_wavelengths,
                 sg_window=5, sg_order=2):
        self.source_wavelengths = np.array(source_wavelengths)
        self.target_wavelengths = np.array(target_wavelengths)

        self.sg_window = sg_window
        self.sg_order = sg_order

    def harmonize(self, pixels):
        # interpolation
        interpolator = interp1d(
            self.source_wavelengths,
            pixels,
            kind="linear",
            axis=0,
            bounds_error=False,
            fill_value="extrapolate"
        )

        return interpolator(self.target_wavelengths).astype(np.float32)


# spectral harmonizer for finetuning
# unify the spectral grid for various downstream sensors
class SpectralHarmonizer_FT:
    def __init__(self, wavelength_source, wavelength_target):

        # Find the closest target band for each source band
        selected_target_indices = []
        selected_source_indices = []
        used_target_bands = set()

        for i, source_wl in enumerate(wavelength_source):
            closest_idx = np.argmin(np.abs(wavelength_target - source_wl))
            if closest_idx not in used_target_bands:  # Ensure unique selection
                selected_target_indices.append(closest_idx)
                selected_source_indices.append(i)
                used_target_bands.add(closest_idx)

        selected_target_indices = np.array(sorted(selected_target_indices))  # Sort indices
        selected_source_indices = np.array(
            sorted(selected_source_indices, key=lambda x: wavelength_source[x]))  # Sort based on wavelength

        # length should be divided by block size 3
        self.selected_target_indices = selected_target_indices[:len(selected_target_indices) -
                                                                (len(selected_target_indices) % 3)]
        self.selected_source_indices = selected_source_indices[:len(selected_source_indices) -
                                                                (len(selected_source_indices) % 3)]
        self.unselected_target_indices = np.setdiff1d(np.arange(len(wavelength_target)), selected_target_indices)

    def harmonize(self, patches):
        if patches.ndim == 3:
            return patches[:, :, self.selected_source_indices]  # inputs are patches
        else:
            return patches[:, self.selected_source_indices]  # inputs are pixels