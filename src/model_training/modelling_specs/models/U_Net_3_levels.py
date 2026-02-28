from tensorflow import keras

from .models_interface import ModelStrategy


class UNet3Levels(ModelStrategy):
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

        outputs = keras.layers.Conv2D(num_classes, 1, activation="softmax")(x)
        return keras.Model(inputs=inputs, outputs=outputs)

    @staticmethod
    def _conv_block(x, filters: int):
        x = keras.layers.Conv2D(filters, 3, activation="relu", padding="same")(x)
        x = keras.layers.Conv2D(filters, 3, activation="relu", padding="same")(x)
        return x

    def _encoder_block(self, x, filters: int):
        conv = self._conv_block(x, filters)
        pool = keras.layers.MaxPooling2D()(conv)
        return conv, pool

    def _decoder_block(self, x, skip, filters: int):
        x = keras.layers.UpSampling2D()(x)
        x = keras.layers.Conv2D(filters, 2, activation="relu", padding="same")(x)
        x = keras.layers.concatenate([skip, x], axis=3)
        x = self._conv_block(x, filters)
        return x