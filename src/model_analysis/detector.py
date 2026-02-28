from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow import keras
import cv2 as cv

from .postprocessing.postprocessing_interface import PostprocessingStrategy
from model_training.data_preprocessing.normalization.normalization_interface import NormalizationStrategy


class Detector:
    """
    Runs inference on a single image and returns detected object positions.
    """

    def __init__(
        self,
        model: keras.Model,
        postprocessing: PostprocessingStrategy,
        normalization: NormalizationStrategy,
        target_shape: tuple = (256, 256),
    ):
        self.model = model
        self.postprocessing = postprocessing
        self.normalization = normalization    # Needed for "from raw"
        self.target_shape = target_shape

    @classmethod
    def from_saved_model(
        cls,
        model_path: Path,
        postprocessing: PostprocessingStrategy,
        normalization: NormalizationStrategy,
        target_shape: tuple = (256, 256),
    ) -> "Detector":

        model = keras.models.load_model(str(model_path), compile=False)
        return cls(model, postprocessing, normalization, target_shape)

    def predict_mask(self, preprocessed_image: np.ndarray) -> np.ndarray:
        tensor = tf.convert_to_tensor([np.expand_dims(preprocessed_image, axis=-1)])
        predicted = self.model.predict(tensor, verbose=0)[0]
        return np.argmax(predicted, axis=-1).astype(np.uint8)

    def predict_from_raw(self, raw_image: np.ndarray) -> tuple[np.ndarray, list[tuple]]:
        normalized = self.normalization.normalize(raw_image)
        resized = cv.resize(normalized, (self.target_shape[1], self.target_shape[0]))
        mask = self.predict_mask(resized)
        positions = self.postprocessing.extract_positions(mask)
        return mask, positions