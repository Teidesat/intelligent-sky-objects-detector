import cv2 as cv
import numpy as np

from .postprocessing_interface import PostprocessingStrategy


class SimpleContour(PostprocessingStrategy):
    """
    Contour detection
    """

    def __init__(self, min_area: float = 10.0):
        self.min_area = min_area

    def extract_positions(self, segmentation_mask: np.ndarray) -> list[tuple]:
        contours, _ = cv.findContours(
            segmentation_mask, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE
        )
        positions = []
        for contour in contours:
            if cv.contourArea(contour) < self.min_area:
                continue
            x_coord, y_coord, width, height = cv.boundingRect(contour)
            positions.append((x_coord + width / 2, y_coord + height / 2))
        return positions