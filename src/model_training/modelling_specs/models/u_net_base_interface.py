from tensorflow import keras
from .models_interface import ModelStrategy


class UNetBase(ModelStrategy):
    """
    Base class for all UNet variants.
    Provides shared encoder/decoder/conv blocks.
    """

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