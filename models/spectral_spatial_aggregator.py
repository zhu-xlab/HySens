import torch.nn as nn


# spectral spatial aggregator, used for patch input
class Spectral_Spatial_Aggregator(nn.Module):
    def __init__(self):
        super(Spectral_Spatial_Aggregator, self).__init__()
        self.conv3d_block = nn.Sequential(
            nn.Conv3d(96, 64, kernel_size=1, stride=1, padding=1),
            nn.BatchNorm3d(64),
            nn.GELU(),

            nn.Conv3d(64, 512, kernel_size=3, stride=(2, 2, 2), padding=1),  # Downsample D, H, W
            nn.BatchNorm3d(512),
            nn.GELU(),

            nn.AdaptiveAvgPool3d((None, 1, 1))  # Only pool over H and W
        )

    def forward(self, x):
        x = x.permute(0, 4, 3, 1, 2)
        x = self.conv3d_block(x)
        x = x.view(x.size(0), -1)
        return x