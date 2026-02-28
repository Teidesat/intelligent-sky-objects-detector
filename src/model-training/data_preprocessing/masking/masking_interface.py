from abc import ABC, abstractmethod

import numpy as np


class MaskingStrategy(ABC):
    """Base class for all segmentation mask creation strategies"""

    @abstractmethod
    def build_mask(self, objects_info: np.ndarray, original_image_shape: tuple, 
                   target_shape: tuple,) -> tuple[np.ndarray, list]:
        """
        Build a binary segmentation mask from detected object info.

        Args:
            objects_info: Array of shape (N, 3) with columns [X, Y, FLUX].
            original_image_shape: (H, W) of the image before resizing.
            target_shape: (H, W) of the resized image the mask must match.

        Returns:
            mask: uint8 array of shape target_shape (0=background, 1=object).
            filtered_objects: list of (x, y, flux) tuples that passed filtering.
        """
        ...