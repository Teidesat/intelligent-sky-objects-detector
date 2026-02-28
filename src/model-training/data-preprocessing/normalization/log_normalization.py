import numpy as np

from .normalization_interface import NormalizationStrategy


class LogPercentileNormalization(NormalizationStrategy):
    """
    Sky background subtraction + logarithmic normalization
    """

    def __init__(self, lowest_percentile: float = 1.0, highest_percentile: float = 99.0):
        self._lowest_percentile = lowest_percentile
        self._highest_percentile = highest_percentile

    def normalize(self, image: np.ndarray) -> np.ndarray:
        background = np.median(image)
        sky_subtracted = np.clip(image.astype(np.float32) - background, 0, None)
        log_image = np.log1p(sky_subtracted)
        lowest_percentile, highest_percentile = np.percentile(log_image, [self._lowest_percentile, self._highest_percentile])
        range_val = max(highest_percentile - lowest_percentile, 1e-6)
        return np.clip((log_image - lowest_percentile) / range_val, 0, 1)