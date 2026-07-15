import json
from pathlib import Path

import astropy.io.fits as fits
import cv2 as cv
import numpy as np

from astropy.utils.data import conf
conf.allow_internet = False

from .entry import DatasetEntry
from .normalization.normalization_interface import NormalizationStrategy
from .masking.masking_interface import MaskingStrategy


class DatasetLoader:
    """
    Loader for datasets structured as directories containing subdirectories for each entry.
    Each entry directory should contain:
    - A FITS image file named "<entry_id>-image.fits"
    - A FITS table file named "<entry_id>-axy.fits" containing detected objects
    - An optional JSON file named "<entry_id>-annotations.json" for additional annotations
    The loader applies a normalization strategy to the images and a masking strategy to generate segmentation masks.
    The resulting DatasetEntry objects are cached as .npy files for faster subsequent loading.
    """
    def __init__(self, normalization: NormalizationStrategy, masking: MaskingStrategy,
                 target_shape: tuple = (256, 256)):
        self.normalization = normalization
        self.masking       = masking
        self.target_shape  = target_shape
        # Cache key: automatically changes when masking strategy changes
        self._cache_key    = type(masking).__name__

    def load(self, dataset_path: Path) -> dict[str, DatasetEntry]:
        cache_dir = dataset_path / "cache"
        cache_dir.mkdir(exist_ok=True)
        dataset = {}

        for entry_path in dataset_path.iterdir():
            if not entry_path.is_dir():
                continue
            entry_id   = entry_path.name
            cache_file = cache_dir / f"{entry_id}_{self._cache_key}.npy"
            ann_path   = entry_path / f"{entry_id}-annotations.json"

            # Use cache if it exists and is not outdated relative to the annotations
            cache_valid = (
                cache_file.exists() and
                not (ann_path.exists() and
                     ann_path.stat().st_mtime > cache_file.stat().st_mtime)
            )

            if cache_valid:
                data = np.load(cache_file, allow_pickle=True).item()
                entry = DatasetEntry(
                    entry_id=data["entry_id"],
                    nn_input_image=data["nn_input_image"],
                    segmentation_mask=data["segmentation_mask"],
                    filtered_objects=data["filtered_objects"],
                )
            else:
                entry = self._load_entry(entry_path)
                if entry is None:
                    continue
                np.save(cache_file, {
                    "entry_id":          entry.entry_id,
                    "nn_input_image":    entry.nn_input_image,
                    "segmentation_mask": entry.segmentation_mask,
                    "filtered_objects":  entry.filtered_objects,
                })

            dataset[entry_id] = entry

        print(f"Dataset loaded: {len(dataset)} entries")
        return dataset

    def _load_entry(self, entry_path: Path) -> DatasetEntry | None:
        entry_id = entry_path.name
        try:
            fits_image       = self._read_fits_image(entry_path, entry_id)
            detected_objects = self._read_detected_objects(entry_path, entry_id)
        except (OSError, ValueError, TypeError) as exc:
            print(f"Skipping {entry_id}: {exc}")
            return None

        monochrome = self._to_monochrome(fits_image, entry_id)
        if monochrome is None:
            return None

        normalized = self.normalization.normalize(monochrome)
        if np.isnan(normalized).any():
            print(f"NaN después de normalizar en {entry_id}, saltando")
            return None

        resized = cv.resize(normalized, (self.target_shape[1], self.target_shape[0]))
        if np.isnan(resized).any():
            print(f"NaN después de resize en {entry_id}, saltando")
            return None

        col_names = detected_objects.columns.names
        x_col    = next((c for c in col_names if c.upper() in
                         ("X", "X_IMAGE", "XWIN_IMAGE", "XPEAK_IMAGE")), None)
        y_col    = next((c for c in col_names if c.upper() in
                         ("Y", "Y_IMAGE", "YWIN_IMAGE", "YPEAK_IMAGE")), None)
        flux_col = next((c for c in col_names if "FLUX" in c.upper()), None)

        if not all([x_col, y_col, flux_col]):
            print(f"Skipping {entry_id}: missing X/Y/FLUX columns.")
            return None

        objects_info = np.stack(
            [detected_objects[x_col], detected_objects[y_col], detected_objects[flux_col]],
            axis=-1,
        )

        annotations = self._read_annotations(entry_path, entry_id)

        mask, filtered_objects = self.masking.build_mask(
            objects_info=objects_info,
            original_image_shape=monochrome.shape,
            target_shape=self.target_shape,
            original_image=monochrome,
            annotations=annotations,
        )

        return DatasetEntry(
            entry_id=entry_id,
            nn_input_image=resized.copy(),
            segmentation_mask=mask.copy(),
            filtered_objects=filtered_objects,
        )

    def _read_annotations(self, entry_path: Path, entry_id: str) -> list[dict] | None:
        ann_path = entry_path / f"{entry_id}-annotations.json"
        if not ann_path.exists():
            return None
        try:
            with open(ann_path) as f:
                return json.load(f)
        except Exception:
            return None

    def _read_fits_image(self, entry_path: Path, entry_id: str) -> np.ndarray:
        image_path = entry_path / f"{entry_id}-image.fits"
        if not image_path.exists():
            raise FileNotFoundError(f"FITS image not found: {image_path}")
        with fits.open(image_path) as hdul:
            data = hdul[0].data
        if data is None:
            raise ValueError("Empty FITS image")
        if np.isnan(data).any():
            data = np.nan_to_num(data, nan=0.0)
        return data

    def _read_detected_objects(self, entry_path: Path, entry_id: str):
        axy_path = entry_path / f"{entry_id}-axy.fits"
        if not axy_path.exists():
            raise FileNotFoundError(f"AXY file not found: {axy_path}")
        with fits.open(axy_path) as hdul:
            data = hdul[1].data
        if data is None or len(data) == 0:
            raise ValueError("Empty or missing objects table")
        return data

    def _to_monochrome(self, fits_image: np.ndarray, entry_id: str) -> np.ndarray | None:
        if len(fits_image.shape) == 2:
            return fits_image.astype(np.float32)
        if len(fits_image.shape) == 3 and fits_image.shape[0] == 3:
            try:
                rgb = np.rollaxis(fits_image, 0, 3)
                return cv.cvtColor(rgb, cv.COLOR_BGR2GRAY).astype(np.float32)
            except cv.error:
                print(f"Warning! Could not convert to monochrome: {entry_id}")
                return None
        print(f"Warning! Unknown image shape {fits_image.shape} at {entry_id}")
        return None