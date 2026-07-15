import numpy as np
import cv2 as cv

from .masking_interface import MaskingStrategy


class VerifiedLocalSNRMasking(MaskingStrategy):
    """
    It only accepts an object from the .axy if in the REAL IMAGE its position is a
    significant local peak (z-score over the local background). 
    If skip_z_check is True, it will include all objects regardless of z-score.
    """
    def __init__(self, window: int = 9, z_threshold: float = 4.0,
                 radius: int = 4, border_margin: int = 5,
                 skip_z_check: bool = False):
        self.window = window
        self.z_threshold = z_threshold
        self.radius = radius
        self.border_margin = border_margin
        self.skip_z_check = skip_z_check

    def build_mask(self, objects_info, original_image_shape, target_shape, original_image=None, annotations=None):
        original_height, original_width = original_image_shape[:2]
        target_height, target_width = target_shape
        half = self.window // 2

        mask = np.zeros(target_shape, dtype=np.uint8)
        filtered_objects = []

        for x_coord, y_coord, flux in objects_info:
            xi, yi = int(round(x_coord)), int(round(y_coord))

            if (xi < self.border_margin or yi < self.border_margin or
                    xi >= original_width - self.border_margin or
                    yi >= original_height - self.border_margin):
                continue

            if not self.skip_z_check:
                y0, y1 = max(0, yi - half), min(original_height, yi + half + 1)
                x0, x1 = max(0, xi - half), min(original_width, xi + half + 1)
                patch = original_image[y0:y1, x0:x1]
                if patch.size < 9:
                    continue
                local_median = np.median(patch)
                local_std = np.std(patch) + 1e-6
                z_score = (original_image[yi, xi] - local_median) / local_std
                if z_score < self.z_threshold:
                    continue

            scaled_x = (x_coord * target_width) / original_width
            scaled_y = (y_coord * target_height) / original_height
            filtered_objects.append((scaled_x, scaled_y, flux))
            cv.circle(mask, (int(round(scaled_x)), int(round(scaled_y))), self.radius, 1, -1)

        return mask, filtered_objects