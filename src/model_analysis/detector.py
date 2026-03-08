from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import cv2 as cv

from .postprocessing.postprocessing_interface import PostprocessingStrategy
from model_training.data_preprocessing.normalization.normalization_interface import NormalizationStrategy


class Detector:
    """
    Runs inference on a single image and returns detected object positions.
    """

    def __init__(
        self,
        model: nn.Module,
        postprocessing: PostprocessingStrategy,
        normalization: NormalizationStrategy,
        target_shape: tuple = (256, 256),
    ):
        self.model = model
        self.postprocessing = postprocessing
        self.normalization = normalization
        self.target_shape = target_shape
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()

    @classmethod
    def from_saved_model(
        cls,
        model_path: Path,
        postprocessing: PostprocessingStrategy,
        normalization: NormalizationStrategy,
        target_shape: tuple = (256, 256),
    ) -> "Detector":
        model = torch.load(str(model_path), map_location="cpu", weights_only=False)
        return cls(model, postprocessing, normalization, target_shape)

    def predict_mask(self, preprocessed_image: np.ndarray) -> np.ndarray:
        # (H, W) → (H, W, 1)
        if preprocessed_image.ndim == 2:
            preprocessed_image = preprocessed_image[:, :, np.newaxis]

        # (H, W, 1) → (1, 1, H, W)
        tensor = torch.tensor(preprocessed_image, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)
        tensor = tensor.to(self.device)

        with torch.no_grad():
            predicted = self.model(tensor)  # (1, C, H, W)

        prob = predicted[0, 0].cpu().numpy()
        binary_mask = (prob > 0.5).astype(np.uint8)
        return binary_mask

    def predict_from_raw(self, raw_image: np.ndarray) -> tuple[np.ndarray, list[tuple]]:
        normalized = self.normalization.normalize(raw_image)
        resized = cv.resize(normalized, (self.target_shape[1], self.target_shape[0]))
        mask = self.predict_mask(resized)
        positions = self.postprocessing.extract_positions(mask)
        return mask, positions