"""
run_evaluation.py — Evaluate a trained model on a test split.

Usage:
    python run_evaluation.py --model ./trained-models/model-xxx.keras --dataset ./dataset/
"""

import argparse
import sys
from pathlib import Path

from sklearn.model_selection import train_test_split

sys.path.append(str(Path(__file__).parent.parent))

from model_training.data_preprocessing.loader import DatasetLoader 
from model_training.data_preprocessing.normalization.log_normalization import LogPercentileNormalization
from model_training.data_preprocessing.masking.circular_dynamic_masking import CircularDynamicMasking
from model_training.data_preprocessing.masking.annotation_masking import AnnotationMasking
from model_analysis.detector import Detector
from model_analysis.evaluator import Evaluator
from model_analysis.postprocessing.morphological_postprocessing import MorphologicalClosing

def parse_args():
    parser = argparse.ArgumentParser(
        prog="run_evaluation.py",
        description=(
            "Evaluate a trained sky object detection model on a test split.\n"
            "Loads the dataset, runs inference on the test subset, and saves\n"
            "visual reports (inference grid, dataset sample) to the output directory."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python run_evaluation.py --model ./trained-models/model-2024.keras --dataset ../../data/dataset/\n"
            "  python run_evaluation.py --model ./trained-models/model-2024.keras --dataset ../../data/dataset/ --output ../../data/performance/ --samples 18\n"
        ),
    )
    parser.add_argument(
        "--model",
        required=True,
        type=Path,
        metavar="PATH",
        help="Path to the trained .keras model file.",
    )
    parser.add_argument(
        "--dataset",
        required=True,
        type=Path,
        metavar="PATH",
        help="Path to the dataset directory containing FITS entries.",
    )
    parser.add_argument(
        "--output",
        default=Path("../../data/performance/"),
        type=Path,
        metavar="PATH",
        help="Directory where output plots will be saved. Default: ../../data/performance/",
    )
    parser.add_argument(
        "--shape",
        default=256,
        type=int,
        metavar="N",
        help="Target image size (square). Must match the size used during training. Default: 256",
    )
    parser.add_argument(
        "--samples",
        default=27,
        type=int,
        metavar="N",
        help="Number of test samples to include in the inference grid plot. Default: 27",
    )
    return parser.parse_args()

def main():
    args = parse_args()

    target_shape   = (args.shape, args.shape)
    normalization  = LogPercentileNormalization()
    masking        = AnnotationMasking(radius=4, border_margin=5)
    postprocessing = MorphologicalClosing()

    load = DatasetLoader(normalization=normalization, masking=masking, target_shape=target_shape)
    print("Loading dataset...")
    dataset = load.load(args.dataset)

    items = list(dataset.items())
    _, rest      = train_test_split(items, train_size=0.7, shuffle=True, random_state=42)
    test_items, _ = train_test_split(rest,  train_size=0.5, shuffle=True, random_state=42)
    test_dataset = dict(test_items)
    print(f"Test set: {len(test_dataset)} entries")

    detector = Detector.from_saved_model(
        model_path=args.model,
        postprocessing=postprocessing,
        normalization=normalization,
        target_shape=target_shape,
    )

    evaluator = Evaluator(detector=detector, output_dir=args.output)
    evaluator.plot_inference_grid(test_dataset, n_samples=args.samples)
    evaluator.compute_object_detection_metrics(test_dataset, tolerance_px=5)
    evaluator.plot_dataset_sample(dataset, n_samples=args.samples)


if __name__ == "__main__":
    main()