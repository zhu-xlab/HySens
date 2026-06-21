import torch.nn as nn


# transformer based spectral encoder
class SpectralEncoder(nn.Module):

    def __init__(
        self,
        hidden_dim=96,
        n_heads=8,
        n_layers=12
    ):
        super().__init__()

        layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=n_heads,
            dim_feedforward=hidden_dim * 4
        )

        self.encoder = nn.TransformerEncoder(
            layer,
            num_layers=n_layers
        )

    def forward(self, x):
        return self.encoder(x)

