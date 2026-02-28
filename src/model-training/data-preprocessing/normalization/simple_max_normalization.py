import numpy as np

from .normalization_interface import NormalizationStrategy

class SimpleMaxNormalization(NormalizationStrategy):
    """
    Divide by global maximum
    """

    def normalize(self, image: np.ndarray) -> np.ndarray:
        max_val = image.max()
        if max_val == 0:
            return image.astype(np.float32)
        return (image / max_val).astype(np.float32)