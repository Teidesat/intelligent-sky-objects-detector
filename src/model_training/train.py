"""
train.py — Entry point for training.

To swap any strategy, change only the instantiation in the Strategies section:

    normalization  = SimpleMaxNormalization()
    loss           = CrossEntropyLoss()
    model_strategy = UNet4Levels()
"""

import sys
import argparse
from pathlib import Path

import numpy as np
import torch
from sklearn.model_selection import train_test_split

sys.path.append(str(Path(__file__).parent.parent.parent / "src"))
sys.stdout.reconfigure(line_buffering=True)

from model_training.data_preprocessing.loader import DatasetLoader
from model_training.data_preprocessing.loader_frigate import FrigatePairLoader

from model_training.data_preprocessing.normalization.log_normalization import LogPercentileNormalization
from model_training.data_preprocessing.normalization.normalization_interface import NormalizationStrategy  # AÑADIDO

from model_training.data_preprocessing.masking.annotation_masking import AnnotationMasking
from model_training.data_preprocessing.masking.verified_local_snr_masking import VerifiedLocalSNRMasking

from model_training.data_preprocessing.augmentation.flip_rotate_augmentation import FlipRotateAugmentation
from model_training.data_preprocessing.augmentation.brightness_contrast_augmentation import BrightnessContrastAugmentation
from model_training.data_preprocessing.augmentation.elastic_augmentation import ElasticAugmentation
from model_training.data_preprocessing.augmentation.compose_augmentation import ComposeAugmentation

from model_training.modelling_specs.losses.combined_loss import CombinedLoss
from model_training.modelling_specs.losses.focal_loss import FocalLoss
from model_training.modelling_specs.losses.dice_loss import DiceLoss
from model_training.modelling_specs.losses.focal_tversky_loss import FocalTverskyLoss
from model_training.modelling_specs.losses.cross_entropy_loss import CrossEntropyLoss
from model_training.modelling_specs.losses.bce_loss import BCELoss
from model_training.modelling_specs.losses.bce_loss import WeightedBCELoss
from model_training.modelling_specs.losses.lovasz_loss import LovaszHingeLoss

from model_training.modelling_specs.models.U_Net_3_levels import UNet3Levels
from model_training.modelling_specs.models.U_Net_4_levels import UNet4Levels

from model_training.trainer import Trainer
from model_analysis.detector import Detector
from model_analysis.evaluator import Evaluator
from model_analysis.postprocessing.morphological_postprocessing import MorphologicalClosing

from model_training.auxiliary_functions import dataset_to_tensors
from model_training.auxiliary_functions import limit_empty_images
from model_training.auxiliary_functions import print_experiment_config

class IdentityNormalization(NormalizationStrategy):
    """Normalization strategy that returns the input image unchanged. Used for Frigate datasets where normalization is not needed."""
    def normalize(self, image):
        return image

def parse_args():
    parser = argparse.ArgumentParser(
        prog="train.py",
        description=(
            "Train a sky object detection model on a dataset of astrometry.net FITS images.\n"
            "Loads the dataset, splits into train/val/test, trains a U-Net segmentation model,\n"
            "saves the best checkpoint and final model, and produces evaluation plots."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python train.py\n"
            "  python train.py --dataset ../../data/dataset/ --epochs 100 --batch-size 8\n"
            "  python train.py --dataset ../../data/dataset/ --shape 512 --output ./my-models/\n"
        ),
    )
    parser.add_argument(
        "--dataset",
        default=Path("../../data/dataset/"),
        type=Path,
        metavar="PATH",
        help="Path to the dataset directory containing FITS entries. Default: ../../data/dataset/",
    )
    parser.add_argument(
        "--output",
        default=Path("../../data/trained-models/"),
        type=Path,
        metavar="PATH",
        help="Directory where trained models and checkpoints will be saved. Default: ../../data/trained-models/",
    )
    parser.add_argument(
        "--performance",
        default=Path("../../data/performance/"),
        type=Path,
        metavar="PATH",
        help="Directory where evaluation plots will be saved. Default: ../../data/performance/",
    )
    parser.add_argument(
        "--shape",
        default=256,
        type=int,
        metavar="N",
        help="Target image size (square). Default: 256",
    )
    parser.add_argument(
        "--batch-size",
        default=6,
        type=int,
        metavar="N",
        help="Training batch size. Default: 6",
    )
    parser.add_argument(
        "--epochs",
        default=100,
        type=int,
        metavar="N",
        help="Number of training epochs. Default: 100",
    )
    return parser.parse_args()

def main():
    args = parse_args()
    TARGET_SHAPE = (args.shape, args.shape)

    # augmentations = [
    #     FlipRotateAugmentation(use_flips=True, use_rotations=True),
    #     BrightnessContrastAugmentation(brightness_range=(0.8, 1.2), contrast_range=(0.8, 1.2)),
    #     ElasticAugmentation(alpha=15.0, sigma=4.0, prob=0.3)  # Opcional, activar con cuidado
    # ]
    
    # Strategies
    normalization         = LogPercentileNormalization(lowest_percentile=1.0, highest_percentile=99.9)
    # masking               = VerifiedLocalSNRMasking(window=9, z_threshold=4.0, radius=4, border_margin=5, skip_z_check=False)
    masking               = AnnotationMasking(radius=6, border_margin=5)
    augmentation_strategy = FlipRotateAugmentation(use_flips=True, use_rotations=True)
    # augmentation_strategy = ComposeAugmentation(augmentations)
    loss                  = CombinedLoss(loss_a=DiceLoss(), loss_b=FocalLoss(alpha=0.25, gamma=2.0), weight_a=0.5)
    model_strategy        = UNet3Levels(prior_prob=0.008)
    postprocessing        = MorphologicalClosing(kernel_size=7, min_area=4.0)    
    # loss                  = FocalTverskyLoss(alpha=0.1, beta=0.9, gamma=1.0)
    # loss = LovaszHingeLoss()
    # loss = CombinedLoss(loss_a=DiceLoss(), loss_b=FocalLoss(alpha=0.25, gamma=2.0), weight_a=0.5)
    # loss = CrossEntropyLoss(num_classes=2, star_class_index=1)
    # loss = WeightedBCELoss(pos_weight=30.0)
    # loss = CombinedLoss(loss_a = WeightedBCELoss(pos_weight=20.0), loss_b = DiceLoss(), weight_a=0.8)
    # loss = WeightedBCELoss(pos_weight=10)

    # normalization         = LogPercentileNormalization(lowest_percentile=1.0, highest_percentile=99.9)
    # masking               = VerifiedLocalSNRMasking(window=9, z_threshold=4.0, radius=4, border_margin=5, skip_z_check=False)
    # augmentation_strategy = FlipRotateAugmentation(use_flips=True, use_rotations=True)
    # model_strategy        = UNet3Levels()
    # postprocessing        = MorphologicalClosing(kernel_size=7, min_area=4.0)

    # Detect if the dataset is in Frigate format (contains .npy files) or standard FITS format. This allows for different loading and preprocessing strategies.
    is_frigate = any(p.suffix == ".npy" for p in args.dataset.iterdir())

    # Load dataset, preparing for frigate-based model 
    if is_frigate:
        loader = FrigatePairLoader(target_shape=TARGET_SHAPE)
        print("Loading Frigate dataset...")
        dataset = loader.load(args.dataset)
        normalization = IdentityNormalization()
        masking = None
    else:
        loader = DatasetLoader(normalization=normalization, masking=masking, target_shape=TARGET_SHAPE)
        print("Loading dataset...")
        dataset = loader.load(args.dataset)

    print_experiment_config(
        Normalization = normalization,
        Loss          = loss,
        Model         = model_strategy,
        Augmentation  = augmentation_strategy,
        Masking       = masking
    )

    # Split
    items = list(dataset.items())
    train_items, rest      = train_test_split(items, train_size=0.7, shuffle=True, random_state=42)
    test_items,  val_items = train_test_split(rest,  train_size=0.5, shuffle=True, random_state=42)

    train_dataset = limit_empty_images(dict(train_items), keep_fraction=0.3)
    test_dataset  = dict(test_items)   
    val_dataset   = dict(val_items)
    print(f"Train: {len(train_dataset)} | Test: {len(test_dataset)} | Val: {len(val_dataset)}")

    def theoretical_max_iou(masks: torch.Tensor) -> float:
        """Function to compute the theoretical maximum IoU given the masks in the validation set.
        With so few positives, the denominator (TP + FP + FN) is dominated by potential"""
        star_px = masks.float().sum().item()
        total_px = masks.numel()
        if total_px == 0:
            print("Densidad positivos: 0.0000% (máscara vacía)")
            return 0.0
        ratio = star_px / total_px
        print(f"Densidad positivos: {ratio:.4%}")
        if ratio == 0.0:
            print("IoU perfecto (0 FP): 0.0000 (no hay positivos)")
            print("IoU con 1% FP adicional: 0.0000")
            print("IoU con 0.5% FP adicional: 0.0000")
            return 0.0
        print(f"IoU perfecto (0 FP): {ratio / (ratio + (1-ratio)*0.0):.4f} → 1.0")
        print(f"IoU con 1% FP adicional: {ratio/(ratio + 0.01):.4f}")
        print(f"IoU con 0.5% FP adicional: {ratio/(ratio + 0.005):.4f}")
        return ratio

    # Tensors
    train_images, train_masks = dataset_to_tensors(train_dataset)
    val_images,   val_masks   = dataset_to_tensors(val_dataset)

    theoretical_max_iou(val_masks)
    
    print("train_images:", train_images.shape)
    print("train_masks:", train_masks.shape)

    print("Valores únicos train_masks:", torch.unique(train_masks))
    print("Pixels estrella train:", train_masks.sum().item())

    print("Valores únicos val_masks:", torch.unique(val_masks))
    print("Pixels estrella val:", val_masks.sum().item())

    images_with_stars = (train_masks.sum(dim=(1,2)) > 0).sum()

    print(
        f"Imágenes con estrellas: "
        f"{images_with_stars}/{len(train_masks)}"
    )

    train_star_px = train_masks.float().sum().item()
    val_star_px   = val_masks.float().sum().item()
    train_total   = train_masks.shape[0] * args.shape * args.shape
    val_total     = val_masks.shape[0]   * args.shape * args.shape
    print(f"Train star pixels: {train_star_px:.0f} / {train_total} = {train_star_px/train_total:.4%}")
    print(f"Val   star pixels: {val_star_px:.0f}   / {val_total}   = {val_star_px/val_total:.4%}")

    print("Tamaño train:", len(train_images))
    print("Tamaño val:",   len(val_images))
    print("Train mask unique:", train_masks[0].unique().tolist() if len(train_masks) > 0 else [])

    # Train
    trainer = Trainer(
        model_strategy=model_strategy,
        loss_strategy=loss,
        input_shape=(*TARGET_SHAPE, 1),
        output_dir=args.output,
        augmentation_strategy=augmentation_strategy,
        batch_size=args.batch_size,
        epochs=args.epochs,
    )
    trainer.build()
    history = trainer.train(train_images, train_masks, val_images, val_masks)
    trainer.save()

    # Evaluation
    detector = Detector(
        model=trainer.model,
        postprocessing=postprocessing,
        normalization=normalization,
        target_shape=TARGET_SHAPE,
        output_adapter=loss.predictions_to_probability,
    )
    evaluator = Evaluator(detector=detector, output_dir=args.performance)
    evaluator.plot_loss_curves(history)
    evaluator.plot_dataset_sample(dataset)
    evaluator.plot_inference_grid(test_dataset)
    evaluator.compute_object_detection_metrics(test_dataset, tolerance_px=5)


if __name__ == "__main__":
    main()