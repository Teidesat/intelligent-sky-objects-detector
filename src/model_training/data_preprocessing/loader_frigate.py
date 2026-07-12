from pathlib import Path
import numpy as np
from .entry import DatasetEntry

class FrigatePairLoader:
    def __init__(self, target_shape=(256, 256)):
        self.target_shape = target_shape

    def load(self, dataset_path: Path) -> dict[str, DatasetEntry]:
        dataset = {}
        for img_path in dataset_path.glob("*_image.npy"):
            entry_id = img_path.stem.replace("_image", "")
            mask_path = dataset_path / f"{entry_id}_mask.npy"
            if not mask_path.exists():
                continue
            img = np.load(img_path)
            mask = np.load(mask_path)
            dataset[entry_id] = DatasetEntry(
                entry_id=entry_id,
                nn_input_image=img,
                segmentation_mask=mask,
                filtered_objects=[]  # no .axy
            )
        print(f"Loaded {len(dataset)} Frigate pairs")
        return dataset