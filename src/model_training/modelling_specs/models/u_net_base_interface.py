import torch
import torch.nn as nn
from .models_interface import ModelStrategy


class ConvBlock(nn.Module):
    """
    This block consists of two convolutional layers, each followed by a normalization layer and a ReLU activation.
    The normalization can be either Group Normalization or Batch Normalization, depending on the 'norm' parameter.
    """
    def __init__(self, in_channels: int, out_channels: int, norm: str = "group"):
        super().__init__()

        def make_norm(c):
            if norm == "group":
                return nn.GroupNorm(min(8, c), c)
            return nn.BatchNorm2d(c)

        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            make_norm(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            make_norm(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class EncoderBlock(nn.Module):
    """
    This block consists of a ConvBlock followed by a MaxPooling layer. It is used in the encoder part of the U-Net architecture.
    The ConvBlock extracts features from the input, and the MaxPooling layer reduces the spatial dimensions
    """
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = ConvBlock(in_channels, out_channels)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        conv = self.conv(x)
        pool = self.pool(conv)
        return conv, pool


class DecoderBlock(nn.Module):
    """
    This block consists of an upsampling layer (ConvTranspose2d) followed by a ConvBlock. 
    It is used in the decoder part of the U-Net architecture.
    The upsampling layer increases the spatial dimensions of the input, and the ConvBlock refines the features. 
    The skip connection from the encoder is concatenated with the upsampled features before passing through the ConvBlock.
    """
    def __init__(self, in_channels: int, skip_channels: int, out_channels: int):
        super().__init__()
        self.upsample = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv_block = ConvBlock(out_channels + skip_channels, out_channels)

    def forward(self, x, skip):
        x = self.upsample(x)
        if x.shape[2:] != skip.shape[2:]:
            x = x[:, :, :skip.shape[2], :skip.shape[3]]
        x = torch.cat([skip, x], dim=1)
        return self.conv_block(x)


class UNetBase(ModelStrategy):
    """Base class for all UNet variants."""
    pass