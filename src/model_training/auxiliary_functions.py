import numpy as np
import torch


def dataset_to_tensors(dataset: dict) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Convert a dataset dict of DatasetEntry objects into PyTorch tensors.
    Images: (B, H, W, 1) float32
    Masks:  (B, H, W)    int32
    """
    images, masks = [], []
    for entry in dataset.values():
        images.append(np.expand_dims(entry.nn_input_image, axis=-1))
        masks.append(entry.segmentation_mask)

    return (
        torch.tensor(np.stack(images), dtype=torch.float32),
        torch.tensor(np.stack(masks),  dtype=torch.long),
    )