import torch
import torch.nn as nn


# wavelength based spectral embedding layers (pretraining)
class WavelengthSpectralEmbedding(nn.Module):

    def __init__(
        self,
        input_dim=210,
        block_size=3,
        hidden_dim=96
    ):
        super().__init__()

        self.input_dim = input_dim
        self.block_size = block_size
        self.hidden_dim = hidden_dim

        self.num_blocks = input_dim // block_size

        self.block_embeddings = nn.ModuleList([
            nn.Linear(block_size, hidden_dim)
            for _ in range(self.num_blocks)
        ])

    def forward(self, x):
        batch_size, num_blocks, block_size, num_pixels = x.shape
        # Apply per-block embedding
        embedded = []
        for block_idx in range(self.num_blocks):
            block = x[:, block_idx]
            block = block.permute(0, 2, 1).reshape(-1, self.block_size)
            block_emb = self.block_embeddings[block_idx](block)
            block_emb = block_emb.view(batch_size, num_pixels, self.hidden_dim).permute(1, 0, 2)
            embedded.append(block_emb)

        x = torch.stack(embedded, dim=0)  # [num_blocks, num_pixels, batch_size, hidden_dim]
        x = x.permute(0, 2, 1, 3).reshape(self.num_blocks, batch_size * num_pixels, self.hidden_dim)
        return x


# wavelength based spectral embedding (finetuning)
class WavelengthSpectralEmbedding_FT(nn.Module):

    def __init__(
        self,
        ref_dim=210,
        input_dim=42,
        block_size=3,
        hidden_dim=96,
        wavelengths_group_indices=None,
    ):
        super().__init__()

        self.ref_dim = ref_dim
        self.input_dim = input_dim
        self.block_size = block_size
        self.hidden_dim = hidden_dim
        self.group_indices = wavelengths_group_indices

        self.ref_num_blocks = ref_dim // block_size
        self.num_blocks = input_dim // block_size

        self.block_embeddings = nn.ModuleList([
            nn.Linear(block_size, hidden_dim)
            for _ in range(self.ref_num_blocks)
        ])

    def forward(self, x):
        # Apply per-block embedding
        embedded = []
        for block_idx in range(0, self.num_blocks, 1):
            block = x[:, block_idx]
            emb_idx = self.group_indices[block_idx]
            block_emb = self.block_embeddings[emb_idx](block)
            embedded.append(block_emb)

        x = torch.stack(embedded, dim=0)
        return x