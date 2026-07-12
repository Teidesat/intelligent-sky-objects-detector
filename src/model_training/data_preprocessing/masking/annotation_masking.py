import numpy as np
import cv2 as cv

from .masking_interface import MaskingStrategy


class AnnotationMasking(MaskingStrategy):
    """
    Ground truth basado en objetos verificados contra catálogos astronómicos
    reales (HD, Tycho-2, 2MASS, USNO-B) devueltos por la API de astrometry.net.
    Ignora completamente el .axy (ruidoso).
    Si no hay annotations para un job, devuelve máscara vacía.
    """

    def __init__(self, radius: int = 4, border_margin: int = 5):
        self.radius = radius
        self.border_margin = border_margin

    def build_mask(
        self,
        objects_info: np.ndarray,
        original_image_shape: tuple,
        target_shape: tuple,
        original_image: np.ndarray | None = None,
        annotations: list[dict] | None = None,
    ) -> tuple[np.ndarray, list]:

        mask     = np.zeros(target_shape, dtype=np.uint8)
        filtered = []

        if not annotations:
            return mask, filtered

        oh, ow = original_image_shape[:2]
        th, tw = target_shape

        for ann in annotations:
            x = float(ann.get("pixelx", -1))
            y = float(ann.get("pixely", -1))
            if x < 0 or y < 0:
                continue
            if (x < self.border_margin or y < self.border_margin or
                    x >= ow - self.border_margin or y >= oh - self.border_margin):
                continue

            sx = (x * tw) / ow
            sy = (y * th) / oh
            filtered.append((sx, sy, 1.0))
            cv.circle(mask, (int(round(sx)), int(round(sy))), self.radius, 1, -1)

        return mask, filtered