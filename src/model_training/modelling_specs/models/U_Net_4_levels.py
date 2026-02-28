from tensorflow import keras

from .u_net_base_interface import UNetBase

class UNet4Levels(UNetBase):
    """
    Original U-Net with 4 encoder/decoder levels.

    Higher capacity than UNet3Level but stronger bottleneck.
    Strong bottlenecks could vanish small points
    """

    def build(self, input_shape: tuple, num_classes: int) -> keras.Model:
        inputs = keras.Input(shape=input_shape)

        conv1, pool1 = self._encoder_block(inputs, 64)
        conv2, pool2 = self._encoder_block(pool1, 128)
        conv3, pool3 = self._encoder_block(pool2, 256)
        conv4, pool4 = self._encoder_block(pool3, 512)

        bridge = self._conv_block(pool4, 1024)

        x = self._decoder_block(bridge, conv4, 512)
        x = self._decoder_block(x, conv3, 256)
        x = self._decoder_block(x, conv2, 128)
        x = self._decoder_block(x, conv1, 64)

        outputs = keras.layers.Conv2D(num_classes, 1, activation="softmax")(x)
        return keras.Model(inputs=inputs, outputs=outputs)