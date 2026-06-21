import torch.nn as nn
import torch
import numpy as np


# Hysens encoder for pretraining
class HysensEncoder(nn.Module):

    def __init__(
        self,
        embedding_layer,
        pe,
        encoder,
        hidden_dim=96,
        block_size=3,
        num_blocks=70,
    ):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.block_size = block_size
        self.num_blocks = num_blocks
        hs_wavelength_grid = np.arange(400, 2500 + 1, 10, dtype=np.float64)
        self.wavelengths = hs_wavelength_grid[0:210:1]

        # wavelength based spectral embedding
        self.embedding = embedding_layer

        # learnable mask tokens
        self.mask_tokens = nn.Parameter(torch.randn(self.num_blocks, self.hidden_dim))

        # positional encoding
        self.pe = pe
        self.absolute_embs = self.pe(torch.tensor(self.wavelengths, dtype=torch.float32),
                                                       self.block_size)

        # spectral encoder
        self.encoder = encoder

    def mask_embeddings(self, embeddings, mask_ratio):
        """
        Masks a fraction of the embeddings by replacing them with the learnable mask token.
        """
        batch_size = embeddings.shape[1]
        num_blocks = embeddings.shape[0]
        mask = torch.rand(num_blocks, batch_size) < mask_ratio

        # Clone the embeddings to avoid in-place operations
        masked_embeddings = embeddings.clone()

        # Iterate over blocks and apply mask tokens where mask is True
        for block_idx in range(num_blocks):
            masked_embeddings[block_idx][mask[block_idx]] = self.mask_tokens[block_idx]

        return masked_embeddings, mask

    def forward(self, x, mask_ratio=0.8):
        batch_size, input_dim, num_pixels = x.shape
        num_blocks = input_dim // self.block_size
        x = x.view(batch_size, num_blocks, self.block_size, num_pixels)

        x = self.embedding(x)
        x = x + self.absolute_embs.unsqueeze(1)
        x, mask = self.mask_embeddings(x, mask_ratio)

        x = self.encoder(x)

        return x


# Hysens encoder for finetuning (the input is hyperspectral patch)
class HysensEncoder_FT(nn.Module):

    def __init__(
        self,
        embedding_layer,
        pe,
        encoder,
        hidden_dim=96,
        block_size=3,
        num_blocks=70,
        wavelengths_group_indices=None,
    ):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.block_size = block_size
        self.num_blocks = num_blocks

        # wavelength based spectral embedding
        self.embedding = embedding_layer

        # positional encoding
        self.pe = pe
        self.absolute_embs = self.pe(wavelengths_group_indices)

        # spectral encoder
        self.encoder = encoder

    def forward(self, x):
        batch_size, patch_size, _, input_dim = x.shape
        num_blocks = input_dim // self.block_size
        x = x.view(batch_size * patch_size * patch_size, input_dim)
        x = x.view(-1, num_blocks, self.block_size)

        x = self.embedding(x)
        x = x + self.absolute_embs.unsqueeze(1)
        x = self.encoder(x)

        return x


# Hysens encoder for finetuning (the input is hyperspectral pixel)
class HysensEncoder_FT_Pixel(nn.Module):

    def __init__(
        self,
        embedding_layer,
        pe,
        encoder,
        hidden_dim=96,
        block_size=3,
        num_blocks=70,
        wavelengths_group_indices=None,
    ):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.block_size = block_size
        self.num_blocks = num_blocks

        # wavelength based spectral embedding
        self.embedding = embedding_layer

        # positional encoding
        self.pe = pe
        self.absolute_embs = self.pe(wavelengths_group_indices)

        # spectral encoder
        self.encoder = encoder

    def forward(self, x):
        batch_size, input_dim = x.shape
        num_blocks = input_dim // self.block_size
        x = x.view(batch_size, input_dim)
        x = x.view(-1, num_blocks, self.block_size)

        x = self.embedding(x)
        x = x + self.absolute_embs.unsqueeze(1)
        x = self.encoder(x)

        return x