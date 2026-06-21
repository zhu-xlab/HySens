import torch.nn as nn
import torch


# the pretrianing framework
class HysensPretrainModel(nn.Module):

    def __init__(
        self,
        encoder,
        decoder
    ):
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder

    def forward(self, x, mask_ratio=0.8):
        batch_size, input_dim, num_pixels = x.shape
        num_blocks = input_dim // 3
        x = self.encoder(x, mask_ratio=mask_ratio)
        x_recon = self.decoder(x)
        x_recon = x_recon.view(num_blocks, batch_size, num_pixels, 3).permute(1, 0, 3, 2).contiguous().view(
            batch_size, input_dim, num_pixels)
        return x_recon


# classification model
class HysensClassificationModel(nn.Module):

    def __init__(
        self,
        encoder,
        aggregator,
        num_outputs=2,
        batch_size=16,
        patch_size=8,
        num_blocks=14,
        hidden_dim=96,
        input_dim=42,
    ):
        super().__init__()

        self.encoder = encoder
        self.aggregator = aggregator
        self.batch_size = batch_size
        self.patch_size = patch_size
        self.num_blocks = num_blocks
        self.hidden_dim = hidden_dim
        self.input_dim = input_dim

        self.Flatten = nn.Flatten()
        self.linear1 = nn.LazyLinear(1024)
        self.linear2 = nn.Linear(1024, 1024)
        self.linear3 = nn.Linear(1024, num_outputs)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(0.2)
        self.dropout1 = nn.Dropout(0.4)

    def forward(self, x):
        # extract features using the pretrained model
        x = self.encoder(x)
        x = x.permute(1, 0, 2)

        x = x.view(self.batch_size, self.patch_size, self.patch_size, self.num_blocks, self.hidden_dim)
        x = self.aggregator(x)

        # pass the features through the linear layers
        x = self.Flatten(x)
        x = self.linear1(x)
        x = self.dropout(x)
        x = self.act(x)
        x = self.linear2(x)
        x = self.dropout1(x)
        x = self.act(x)
        x = self.linear3(x)
        x = self.act(x)
        return x


# change detection model
class HysensChangeDetectionModel(nn.Module):

    def __init__(
        self,
        encoder,
        aggregator,
        batch_size=16,
        patch_size=8,
        num_blocks=14,
        hidden_dim=96,
        input_dim=42,
    ):
        super().__init__()

        self.encoder = encoder
        self.aggregator = aggregator
        self.batch_size = batch_size
        self.patch_size = patch_size
        self.num_blocks = num_blocks
        self.hidden_dim = hidden_dim
        self.input_dim = input_dim

        self.Flatten = nn.Flatten()
        self.linear1 = nn.LazyLinear(1024)
        self.linear2 = nn.Linear(1024, 1024)
        self.linear3 = nn.Linear(1024, 2)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(0.2)
        self.dropout1 = nn.Dropout(0.4)

    def forward(self, x_pre, x_post):
        # extract features using the pretrained model
        x_pre = self.encoder(x_pre)
        x_post = self.encoder(x_post)
        x_pre = x_pre.permute(1, 0, 2)
        x_post = x_post.permute(1, 0, 2)
        x_pre = x_pre.view(self.batch_size, self.patch_size, self.patch_size, self.num_blocks, self.hidden_dim)
        x_post = x_post.view(self.batch_size, self.patch_size, self.patch_size, self.num_blocks, self.hidden_dim)
        x_diff = x_post - x_pre

        x = self.aggregator(x_diff)

        # pass the features through the linear layers
        x = self.Flatten(x)
        x = self.linear1(x)
        x = self.dropout(x)
        x = self.act(x)
        x = self.linear2(x)
        x = self.dropout1(x)
        x = self.act(x)
        x = self.linear3(x)
        x = self.act(x)
        return x


# simple encoder for SAR data
class SimpleSAREncoder(nn.Module):
    def __init__(self, in_channels=4, out_dim=64):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),

            nn.AdaptiveAvgPool2d(1)
        )
        self.fc = nn.Linear(64, out_dim)

    def forward(self, x):
        x = x.permute(0, 3, 1, 2)
        x = self.encoder(x)
        x = x.flatten(1)
        return self.fc(x)


# modality fusion model of HS and SAR
class HysensFusionModel(nn.Module):

    def __init__(
        self,
        encoder,
        aggregator,
        num_outputs=2,
        batch_size=16,
        patch_size=8,
        num_blocks=14,
        hidden_dim=96,
        input_dim=42,
    ):
        super().__init__()

        self.batch_size = batch_size
        self.patch_size = patch_size
        self.num_blocks = num_blocks
        self.hidden_dim = hidden_dim
        self.input_dim = input_dim

        self.encoder = encoder
        self.aggregator = aggregator
        self.SAR_encoder = SimpleSAREncoder()

        # hs projection
        self.hs_proj = nn.Sequential(
            nn.LazyLinear(128),
            nn.BatchNorm1d(128),
            nn.ReLU()
        )

        # SAR projection
        self.sar_proj = nn.Sequential(
            nn.LazyLinear(128),
            nn.BatchNorm1d(128),
            nn.ReLU()
        )

        self.Flatten = nn.Flatten()
        self.linear1 = nn.LazyLinear(1024)
        self.linear2 = nn.Linear(1024, 1024)
        self.linear3 = nn.Linear(1024, num_outputs)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(0.2)
        self.dropout1 = nn.Dropout(0.4)

    def forward(self, x_hs, x_sar):
        # extract features using the pretrained transformer model
        x_hs = self.encoder(x_hs)
        x_hs = x_hs.permute(1, 0, 2)
        x_hs = x_hs.view(self.batch_size, self.patch_size, self.patch_size, self.num_blocks, self.hidden_dim)
        x_hs = self.aggregator(x_hs)

        x_sar = self.SAR_encoder(x_sar)

        x_hs = self.hs_proj(x_hs)
        x_sar = self.sar_proj(x_sar)
        x = torch.cat([x_hs, x_sar], dim=1)

        # pass the features through the linear layers
        x = self.Flatten(x)
        x = self.linear1(x)
        x = self.dropout(x)
        x = self.act(x)
        x = self.linear2(x)
        x = self.dropout1(x)
        x = self.act(x)
        x = self.linear3(x)
        x = self.act(x)
        return x


# regression model for spectroscopy applications
class HysensRegressionModel(nn.Module):

    def __init__(
        self,
        encoder,
        batch_size=16,
        num_blocks=14,
        hidden_dim=96,
        input_dim=42,
    ):
        super().__init__()

        self.encoder = encoder
        self.batch_size = batch_size
        self.num_blocks = num_blocks
        self.hidden_dim = hidden_dim
        self.input_dim = input_dim

        self.Flatten = nn.Flatten()
        self.linear1 = nn.LazyLinear(128)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(0.25)
        self.linear2 = nn.Linear(128, 32)
        self.linear3 = nn.Linear(32, 1)

    def forward(self, x):
        # extract features using the pretrained model
        x = self.encoder(x)
        x = x.permute(1, 0, 2)

        x = x.view(self.batch_size, self.num_blocks, self.hidden_dim)

        # pass the features through the linear layers
        x = self.Flatten(x)
        x = self.linear1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.linear2(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.linear3(x)
        return x