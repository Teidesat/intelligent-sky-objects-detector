import numpy as np
import cv2 as cv
from scipy.spatial.distance import cdist

from .masking_interface import MaskingStrategy


class CircularDynamicMasking(MaskingStrategy):
    """
      - Flux threshold 
      - Merges nearby detections, should avoid multiple detections of the same bright point
      - Circles scaled by flux // Maybe would be better to sclae with shape, testing needed
    """

    def __init__(self, flux_percentile: float = 30.0, merge_radius: int = 8, min_radius: int = 3, max_radius: int = 12,):
        self.flux_percentile = flux_percentile
        self.merge_radius = merge_radius
        self.min_radius = min_radius
        self.max_radius = max_radius

    def build_mask(self, objects_info: np.ndarray, original_image_shape: tuple, 
                   target_shape: tuple,) -> tuple[np.ndarray, list]:
        
        flux_threshold = np.percentile(objects_info[:, 2], self.flux_percentile)

        adapted = self._adapt_and_filter(objects_info, original_image_shape, target_shape, flux_threshold)
        merged = self._merge_nearby(adapted)

        mask = np.zeros(target_shape, dtype=np.uint8)
        filtered_objects = []

        if not merged:
            return mask, filtered_objects

        max_flux = max(o[2] for o in merged)
        for x_coord, y_coord, flux in merged:
            normalized_flux = np.log1p(flux) / np.log1p(max_flux + 1e-8)
            radius = int(self.min_radius + (self.max_radius - self.min_radius) * normalized_flux)
            filtered_objects.append((x_coord, y_coord, flux))
            cv.circle(mask, (int(x_coord), int(y_coord)), radius, 1, -1)

        return mask, filtered_objects

    def _adapt_and_filter( self, objects_info: np.ndarray, original_shape: tuple, target_shape: tuple, 
                          flux_threshold: float,) -> list:
        
        original_height, original_width = original_shape[:2]
        target_height, target_width = target_shape
        adapted = []
        for x_coord, y_coord, flux in objects_info:
            if flux <= flux_threshold:
                continue
            adapted.append((
                (x_coord * target_width) / original_width,
                (y_coord * target_height) / original_height,
                flux,
            ))
        return adapted

    def _merge_nearby(self, objects: list) -> list:
        if not objects:
            return []
        coords = np.array([(o[0], o[1]) for o in objects])
        fluxes = np.array([o[2] for o in objects])
        distances = cdist(coords, coords)
        merged, used = [], set()
        for i in range(len(objects)):
            if i in used:
                continue
            nearby = [j for j in np.where(distances[i] < self.merge_radius)[0] if j not in used]
            best = nearby[int(np.argmax(fluxes[nearby]))]
            merged.append(objects[best])
            used.update(nearby)
        return merged