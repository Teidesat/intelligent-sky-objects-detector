from tensorflow import keras

from .u_net_base_interface import UNetBase


class UNet3Levels(UNetBase):
    """
    U-Net with 3 encoder/decoder levels.
    """

    def build(self, input_shape: tuple, num_classes: int) -> keras.Model:
        inputs = keras.Input(shape=input_shape)

        # Encoder
        conv1, pool1 = self._encoder_block(inputs, 64)
        conv2, pool2 = self._encoder_block(pool1, 128)
        conv3, pool3 = self._encoder_block(pool2, 256)

        # Bridge
        bridge = self._conv_block(pool3, 512)

        # Decoder
        x = self._decoder_block(bridge, conv3, 256)
        x = self._decoder_block(x, conv2, 128)
        x = self._decoder_block(x, conv1, 64)

        outputs = keras.layers.Conv2D(1, 1, activation="sigmoid")(x)
        return keras.Model(inputs=inputs, outputs=outputs)