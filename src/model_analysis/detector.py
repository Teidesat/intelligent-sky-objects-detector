from pathlib import Path
from typing import Callable

import numpy as np
import torch
import torch.nn as nn
import cv2 as cv

from .postprocessing.postprocessing_interface import PostprocessingStrategy
from model_training.data_preprocessing.normalization.normalization_interface import NormalizationStrategy


def _default_output_adapter(preds: torch.Tensor) -> torch.Tensor:
    """Backward-compatible default output adapter for single-channel Sigmoid outputs. Returns (B, H, W) probability of 'star'."""
    return preds[:, 0]


class Detector:
    """Runs inference on a single image and returns detected object positions."""

    def __init__(
        self,
        model: nn.Module,
        postprocessing: PostprocessingStrategy,
        normalization: NormalizationStrategy,
        target_shape: tuple = (256, 256),
        output_adapter: Callable[[torch.Tensor], torch.Tensor] | None = None,
    ):
        self.model = model
        self.postprocessing = postprocessing
        self.normalization = normalization
        self.target_shape = target_shape
        self.output_adapter = output_adapter or _default_output_adapter
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
        checkpoint = torch.load(str(model_path), map_location="cpu", weights_only=False)

        if isinstance(checkpoint, dict) and "model" in checkpoint:
            model = checkpoint["model"]
            output_channels = checkpoint.get("output_channels", 1)
        else:
            model = checkpoint
            output_channels = 1

        if output_channels == 1:
            output_adapter = _default_output_adapter
        else:
            def output_adapter(preds, _idx=1):
                return torch.softmax(preds, dim=1)[:, _idx]

        return cls(model, postprocessing, normalization, target_shape, output_adapter=output_adapter)

    def predict_mask(self, preprocessed_image: np.ndarray) -> np.ndarray:
        if preprocessed_image.ndim == 2:
            preprocessed_image = preprocessed_image[:, :, np.newaxis]

        tensor = torch.tensor(preprocessed_image, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)
        tensor = tensor.to(self.device)

        with torch.no_grad():
            predicted = self.model(tensor)  # (1, C, H, W)

        prob = self.output_adapter(predicted)[0].cpu().numpy()  # (H, W)
        binary_mask = (prob > 0.3).astype(np.uint8)
        return binary_mask

    def predict_from_raw(self, raw_image: np.ndarray) -> tuple[np.ndarray, list[tuple]]:
        normalized = self.normalization.normalize(raw_image)
        resized = cv.resize(normalized, (self.target_shape[1], self.target_shape[0]))
        mask = self.predict_mask(resized)
        positions = self.postprocessing.extract_positions(mask)
        return mask, positions