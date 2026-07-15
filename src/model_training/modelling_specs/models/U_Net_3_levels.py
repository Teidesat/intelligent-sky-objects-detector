import math
import torch.nn as nn
from .u_net_base_interface import UNetBase, ConvBlock, EncoderBlock, DecoderBlock


class UNet3LevelsModule(nn.Module):
    """
    U-Net with 3 levels. Less capacity than UNet4Levels but a less strong bottleneck.
    """
    def __init__(self, in_channels: int, num_classes: int, apply_activation: bool = True, prior_prob: float = 0.04):
        super().__init__()
        self.enc1 = EncoderBlock(in_channels, 64)
        self.enc2 = EncoderBlock(64, 128)
        self.enc3 = EncoderBlock(128, 256)
        self.bridge = ConvBlock(256, 512)
        self.dec3 = DecoderBlock(512, 256, 256)
        self.dec2 = DecoderBlock(256, 128, 128)
        self.dec1 = DecoderBlock(128, 64, 64)
        self.output_conv = nn.Conv2d(64, num_classes, kernel_size=1)
        self.final_activation = nn.Sigmoid() if apply_activation else nn.Identity()
        self._init_output_bias(prior_prob)

    def _init_output_bias(self, prior_prob: float):
        """
        Initializes the bias of the output convolution layer based on the prior probability. 
        This is important to prevent the model from being biased towards predicting the background class, 
        especially when the foreground class is rare.
        Important to note that this initialization is only applied when the number of output channels is 1 (binary segmentation).
        """
        if self.output_conv.out_channels != 1:
            return  
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        nn.init.constant_(self.output_conv.bias, bias_value)

    def forward(self, x):
        conv1, pool1 = self.enc1(x)
        conv2, pool2 = self.enc2(pool1)
        conv3, pool3 = self.enc3(pool2)
        bridge = self.bridge(pool3)
        x = self.dec3(bridge, conv3)
        x = self.dec2(x, conv2)
        x = self.dec1(x, conv1)
        return self.final_activation(self.output_conv(x))


class UNet3Levels(UNetBase):
    """
    U-Net with 3 levels. Less capacity than UNet4Levels but a less strong bottleneck.
    """
    def __init__(self, prior_prob: float = 0.04):
        self.prior_prob = prior_prob

    def build(self, input_shape: tuple, num_classes: int, apply_activation: bool = True) -> nn.Module:
        in_channels = input_shape[2]
        return UNet3LevelsModule(in_channels=in_channels, num_classes=num_classes,
                                  apply_activation=apply_activation, prior_prob=self.prior_prob)