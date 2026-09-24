from abc import ABC, abstractmethod

import numpy as np


class PostprocessingStrategy(ABC):

    @abstractmethod
    def extract_positions(self, segmentation_mask: np.ndarray) -> list[tuple]:
        """
        Extract (x, y) center positions of detected objects from a binary mask.

        Args:
            segmentation_mask: uint8 array with 0=background, 1=object.

        Returns:
            List of (center_x, center_y) tuples.
        """
        ...