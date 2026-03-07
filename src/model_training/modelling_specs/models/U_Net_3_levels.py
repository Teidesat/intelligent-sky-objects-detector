import torch.nn as nn
from .u_net_base_interface import UNetBase, ConvBlock, EncoderBlock, DecoderBlock


class UNet3LevelsModule(nn.Module):
    def __init__(self, in_channels: int, num_classes: int):
        super().__init__()
        self.enc1 = EncoderBlock(in_channels, 64)
        self.enc2 = EncoderBlock(64, 128)
        self.enc3 = EncoderBlock(128, 256)

        self.bridge = ConvBlock(256, 512)

        self.dec3 = DecoderBlock(512, 256, 256)
        self.dec2 = DecoderBlock(256, 128, 128)
        self.dec1 = DecoderBlock(128, 64, 64)

        self.output_conv = nn.Conv2d(64, 1, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        conv1, pool1 = self.enc1(x)
        conv2, pool2 = self.enc2(pool1)
        conv3, pool3 = self.enc3(pool2)

        bridge = self.bridge(pool3)

        x = self.dec3(bridge, conv3)
        x = self.dec2(x, conv2)
        x = self.dec1(x, conv1)

        return self.sigmoid(self.output_conv(x))


class UNet3Levels(UNetBase):
    """U-Net with 3 encoder/decoder levels."""

    def build(self, input_shape: tuple, num_classes: int) -> nn.Module:
        # input_shape: (H, W, C)
        in_channels = input_shape[2]
        return UNet3LevelsModule(in_channels=in_channels, num_classes=num_classes)