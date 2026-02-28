from abc import ABC, abstractmethod
import numpy as np


class NormalizationStrategy(ABC):
    """Base class for image normalization strategies."""

    @abstractmethod
    def normalize(self, image: np.ndarray) -> np.ndarray:
        """Normalize a monochrome float32 image to [0, 1]."""
        ...