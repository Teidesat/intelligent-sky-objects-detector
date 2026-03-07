import torch
import torch.nn as nn
from .models_interface import ModelStrategy


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class EncoderBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = ConvBlock(in_channels, out_channels)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        conv = self.conv(x)
        pool = self.pool(conv)
        return conv, pool


class DecoderBlock(nn.Module):
    def __init__(self, in_channels: int, skip_channels: int, out_channels: int):
        super().__init__()
        # ConvTranspose2d duplica exactamente el tamaño espacial: H→2H, W→2W
        self.upsample = nn.ConvTranspose2d(
            in_channels, out_channels, kernel_size=2, stride=2
        )
        self.conv_block = ConvBlock(out_channels + skip_channels, out_channels)

    def forward(self, x, skip):
        x = self.upsample(x)                # (B, out_channels, 2H, 2W) — exacto
        if x.shape[2:] != skip.shape[2:]:   # seguridad ante dimensiones impares
            x = x[:, :, :skip.shape[2], :skip.shape[3]]
        x = torch.cat([skip, x], dim=1)     # (B, out_channels + skip_channels, H, W)
        return self.conv_block(x)


class UNetBase(ModelStrategy):
    """Base class for all UNet variants."""
    pass