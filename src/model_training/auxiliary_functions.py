import numpy as np
import torch
import torch.nn as nn
import random

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

def limit_empty_images(dataset: dict, keep_fraction: float = 0.3, seed: int = 42) -> dict:
    """Keeps all images with at least one verified star, and only a fraction of the completely empty ones, 
    so they don't dominate the loss average pushing the model towards 'predict nothing'."""
    rng = random.Random(seed)
    with_stars, empty = {}, {}
    for k, v in dataset.items():
        if v.segmentation_mask.sum() > 0:
            with_stars[k] = v
        else:
            empty[k] = v

    keep_n = int(len(empty) * keep_fraction)
    kept_keys = rng.sample(list(empty.keys()), keep_n) if keep_n < len(empty) else list(empty.keys())
    kept_empty = {k: empty[k] for k in kept_keys}

    print(f"Train: {len(with_stars)} con estrellas + {len(kept_empty)}/{len(empty)} vacías mantenidas")
    return {**with_stars, **kept_empty}

def print_experiment_config(**strategies):
    """
    Prints the configuration of the experiment, including the names and parameters of the provided strategies.
    """
    print("\n" + "="*50)
    print("EXPERIMENT CONFIGURATION")
    print("="*50)

    for name, strategy in strategies.items():
        if strategy is None:
            print(f"{name.upper():<15}: None")
            continue

        class_name = strategy.__class__.__name__
        params = _extract_params(strategy)
        print(f"{name.upper():<15}: {class_name}")
        if params:
            print(f"{' ':<15}  {params}")

    print("="*50 + "\n")


def _extract_params(strategy) -> dict:
    """Extracts printable parameters, with special handling for CombinedLoss."""
    from model_training.modelling_specs.losses.combined_loss import CombinedLoss

    if isinstance(strategy, CombinedLoss):
        return {
            "loss_a":   strategy._reference_strategy.__class__.__name__,
            "loss_b":   strategy.loss_b.__class__.__name__,
            "weight_a": strategy.weight_a,
            "weight_b": strategy.weight_b,
        }

    params = {}
    for k, v in vars(strategy).items():
        if k.startswith('__') or callable(v):
            continue
        if isinstance(v, nn.Module):
            continue
        params[k.lstrip('_')] = v
    return params