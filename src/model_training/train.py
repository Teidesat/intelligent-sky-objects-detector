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
from model_training.data_preprocessing.normalization.log_normalization import LogPercentileNormalization
from model_training.data_preprocessing.normalization.simple_max_normalization import SimpleMaxNormalization
from model_training.data_preprocessing.masking.circular_dynamic_masking import CircularDynamicMasking
from model_training.data_preprocessing.masking.bbox_masking import SimpleBboxMasking
from model_training.modelling_specs.losses.dice_loss import DiceLoss
from model_training.modelling_specs.losses.combined_loss import CombinedLoss
from model_training.modelling_specs.losses.bce_loss import BCELoss
from model_training.modelling_specs.losses.cross_entropy_loss import CrossEntropyLoss
from model_training.modelling_specs.losses.focal_loss import FocalLoss
from model_training.modelling_specs.models.U_Net_3_levels import UNet3Levels
from model_training.modelling_specs.models.U_Net_4_levels import UNet4Levels
from model_training.trainer import Trainer
from model_training.auxiliary_functions import dataset_to_tensors

from model_analysis.detector import Detector
from model_analysis.evaluator import Evaluator
from model_analysis.postprocessing.morphological_postprocessing import MorphologicalClosing


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
        default=50,
        type=int,
        metavar="N",
        help="Number of training epochs. Default: 50",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    TARGET_SHAPE = (args.shape, args.shape)

    # Strategies
    normalization  = LogPercentileNormalization(lowest_percentile=1.0, highest_percentile=99.0)
    masking        = CircularDynamicMasking(flux_percentile=50, merge_radius=8, min_radius=3, max_radius=8)
    loss           = CombinedLoss(loss_a=DiceLoss(), loss_b=FocalLoss(alpha=0.25, gamma=2.0), weight_a=0.5)
    # CombinedLoss(loss_a=DiceLoss(), loss_b=FocalLoss(alpha=0.25, gamma=2.0), weight_a=0.5)
    model_strategy = UNet3Levels()
    postprocessing = MorphologicalClosing(kernel_size=7, min_area=4.0)

    # Load dataset
    loader = DatasetLoader(normalization=normalization, masking=masking, target_shape=TARGET_SHAPE)
    print("Loading dataset...")
    dataset = loader.load(args.dataset)

    # Split
    items = list(dataset.items())
    train_items, rest      = train_test_split(items, train_size=0.5, shuffle=True, random_state=42)
    test_items,  val_items = train_test_split(rest,  train_size=0.5, shuffle=True, random_state=42)

    train_dataset = dict(train_items)
    test_dataset  = dict(test_items)
    val_dataset   = dict(val_items)

    print(f"Train: {len(train_dataset)} | Test: {len(test_dataset)} | Val: {len(val_dataset)}")

    # Tensors
    train_images, train_masks = dataset_to_tensors(train_dataset)
    val_images,   val_masks   = dataset_to_tensors(val_dataset)

    train_star_px = train_masks.float().sum().item()
    val_star_px   = val_masks.float().sum().item()
    train_total   = train_masks.shape[0] * args.shape * args.shape
    val_total     = val_masks.shape[0]   * args.shape * args.shape
    print(f"Train star pixels: {train_star_px:.0f} / {train_total} = {train_star_px/train_total:.4%}")
    print(f"Val   star pixels: {val_star_px:.0f}   / {val_total}   = {val_star_px/val_total:.4%}")

    print("Tamaño train:", len(train_images))
    print("Tamaño val:",   len(val_images))
    print("Train mask unique:", train_masks[0].unique().tolist())

    # Train
    trainer = Trainer(
        model_strategy=model_strategy,
        loss_strategy=loss,
        input_shape=(*TARGET_SHAPE, 1),
        output_dir=args.output,
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
    )
    evaluator = Evaluator(detector=detector, output_dir=args.performance)
    evaluator.plot_loss_curves(history)
    evaluator.plot_dataset_sample(dataset)
    evaluator.plot_inference_grid(test_dataset)


if __name__ == "__main__":
    main()