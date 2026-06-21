import torch
import torch.nn as nn
import numpy as np


# wavelength based PE (pretraining)
class SpectralPositionalEncoding(nn.Module):
    def __init__(self, dim):
        super(SpectralPositionalEncoding, self).__init__()
        self.dim = dim

    def forward(self, wavelengths, block_size):
        position = torch.arange(0, wavelengths.size(0)/block_size, dtype=torch.float32).unsqueeze(1)

        div_term = torch.exp(torch.arange(0, self.dim, 2, dtype=torch.float32) * -(np.log(10000.0) / self.dim))

        sinusoid_inp = position * div_term
        sinusoidal_embedding = torch.cat([torch.sin(sinusoid_inp), torch.cos(sinusoid_inp)], dim=1)

        return sinusoidal_embedding.to('cuda')


# wavelength based PE (finetuning)
class SpectralPositionalEncoding_FT(nn.Module):
    def __init__(self, dim):
        super(SpectralPositionalEncoding_FT, self).__init__()
        self.dim = dim

    def forward(self, wavelengths_group_indices):
        wavelengths_group_indices = torch.tensor(wavelengths_group_indices, dtype=torch.float32)
        closest_indices = wavelengths_group_indices.float().unsqueeze(1)

        div_term = torch.exp(
            torch.arange(0, self.dim, 2, dtype=torch.float32, device=wavelengths_group_indices.device) * -(
                    np.log(10000.0) / self.dim))
        sinusoid_inp = closest_indices * div_term
        sinusoidal_embedding = torch.cat([torch.sin(sinusoid_inp), torch.cos(sinusoid_inp)], dim=1)

        return sinusoidal_embedding.to('cuda')