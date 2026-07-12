from abc import ABC, abstractmethod
import numpy as np


class MaskingStrategy(ABC):

    @abstractmethod
    def build_mask(
        self,
        objects_info: np.ndarray,
        original_image_shape: tuple,
        target_shape: tuple,
        original_image: np.ndarray | None = None,
        annotations: list[dict] | None = None,
    ) -> tuple[np.ndarray, list]:
        """
        Args:
            objects_info:         (N, 3) [X, Y, FLUX] del .axy.
            original_image_shape: (H, W) antes de resize.
            target_shape:         (H, W) destino.
            original_image:       float32 (H, W), opcional.
            annotations:          lista de dicts de astrometry.net/api/jobs/ID/annotations/
                                  ya filtrados a STAR_TYPES. Opcional.
        Returns:
            mask:             uint8 (target_shape), 0=fondo 1=estrella.
            filtered_objects: lista de (x, y, flux).
        """
        ...