import torch
import torch.nn as nn
from .u_net_base_interface import UNetBase, ConvBlock, EncoderBlock, DecoderBlock


class UNet4LevelsModule(nn.Module):
    def __init__(self, in_channels: int, num_classes: int):
        super().__init__()
        self.enc1 = EncoderBlock(in_channels, 64)
        self.enc2 = EncoderBlock(64, 128)
        self.enc3 = EncoderBlock(128, 256)
        self.enc4 = EncoderBlock(256, 512)

        self.bridge = ConvBlock(512, 1024)

        self.dec4 = DecoderBlock(1024, 512, 512)
        self.dec3 = DecoderBlock(512, 256, 256)
        self.dec2 = DecoderBlock(256, 128, 128)
        self.dec1 = DecoderBlock(128, 64, 64)

        self.output_conv = nn.Conv2d(64, 1, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        conv1, pool1 = self.enc1(x)
        conv2, pool2 = self.enc2(pool1)
        conv3, pool3 = self.enc3(pool2)
        conv4, pool4 = self.enc4(pool3)

        bridge = self.bridge(pool4)

        x = self.dec4(bridge, conv4)
        x = self.dec3(x, conv3)
        x = self.dec2(x, conv2)
        x = self.dec1(x, conv1)

        return self.sigmoid(self.output_conv(x))


class UNet4Levels(UNetBase):
    """
    Original U-Net with 4 encoder/decoder levels.
    Higher capacity than UNet3Levels but with a stronger bottleneck.
    """

    def build(self, input_shape: tuple, num_classes: int) -> nn.Module:
        in_channels = input_shape[2]
        return UNet4LevelsModule(in_channels=in_channels, num_classes=num_classes)