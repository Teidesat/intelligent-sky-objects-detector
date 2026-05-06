import cv2 as cv
import numpy as np

from .postprocessing_interface import PostprocessingStrategy


class MorphologicalClosing(PostprocessingStrategy):
    """
    Morphological closing before contour detection.

    Elliptical kernel fusing nearby regions before extracting centers
    """

    def __init__(self, kernel_size: int = 7, min_area: float = 4.0):
        self.kernel_size = kernel_size
        self.min_area = min_area

    def extract_positions(self, segmentation_mask: np.ndarray) -> list[tuple]:
        kernel = cv.getStructuringElement(
            cv.MORPH_ELLIPSE, (self.kernel_size, self.kernel_size)
        )
        closed = cv.morphologyEx(segmentation_mask, cv.MORPH_CLOSE, kernel)
        contours, _ = cv.findContours(closed, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)

        positions = []
        for contour in contours:
            if cv.contourArea(contour) < self.min_area:
                continue
            # Centroide por distancia euclídea, media artimética
            M = cv.moments(contour)
            if M['m00'] == 0:
                continue
            cx = M['m10'] / M['m00']
            cy = M['m01'] / M['m00']
            positions.append((cx, cy))

        return positions