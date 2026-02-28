from math import floor, ceil
import numpy as np

from .masking_interface import MaskingStrategy


class SimpleBboxMasking(MaskingStrategy):
    """
    Bounding-box masking with fixed radius and absolute flux threshold.
    """

    def __init__(self, flux_threshold: float = 70.0, radius: int = 2) -> None:
        self.flux_threshold = flux_threshold
        self.radius = radius

    def build_mask( self, objects_info: np.ndarray, original_image_shape: tuple, 
                   target_shape: tuple, ) -> tuple[np.ndarray, list]:

        original_height, original_width = original_image_shape[:2]
        target_height, target_width = target_shape

        segmentation_mask = np.zeros(target_shape, dtype=np.uint8)
        filtered_objects_info = []

        for x_coord, y_coord, flux in objects_info:

            if flux <= self.flux_threshold:
                continue

            scaled_x = (x_coord * target_width) / original_width
            scaled_y = (y_coord * target_height) / original_height

            filtered_objects_info.append((scaled_x, scaled_y, flux))

            x_min = floor(max(scaled_x - self.radius, 0))
            x_max = ceil(min(scaled_x + self.radius, target_width))

            y_min = floor(max(scaled_y - self.radius, 0))
            y_max = ceil(min(scaled_y + self.radius, target_height))

            segmentation_mask[y_min:y_max, x_min:x_max] = 1

        return segmentation_mask, filtered_objects_info