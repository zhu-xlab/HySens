import torch
import torch.nn as nn


class SpectralDecoder(nn.Module):
    def __init__(
        self,
        input_dim=210,
        block_size=3,
        hidden_dim=96,
    ):
        super().__init__()

        self.input_dim = input_dim
        self.block_size = block_size
        self.hidden_dim = hidden_dim

        self.num_blocks = input_dim // block_size

        self.block_decoders = nn.ModuleList([
            nn.Linear(hidden_dim, block_size)
            for _ in range(self.num_blocks)
        ])

    def forward(self, x):
        # Decode each block separately using corresponding block decoder
        decoded_blocks = []
        for block_idx in range(self.num_blocks):
            block_feat = x[block_idx]  # [batch_size * num_pixels, hidden_dim]
            block_decoded = self.block_decoders[block_idx](block_feat)  # [batch_size * num_pixels, block_size]
            decoded_blocks.append(block_decoded.unsqueeze(0))  # [1, batch_size * num_pixels, block_size]

        x = torch.cat(decoded_blocks, dim=0)  # [num_blocks, batch_size * num_pixels, block_size]
        return x

