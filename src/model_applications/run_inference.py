"""
run_inference.py — Run inference on a single FITS image.

Usage:
    python run_inference.py --model ./trained-models/model-xxx.keras --image ./path/to/image.fits
"""

import argparse
import sys
from pathlib import Path

import astropy.io.fits as fits
import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import PowerNorm

sys.path.append(str(Path(__file__).parent.parent))

from model_training.data_preprocessing.normalization.log_normalization import LogPercentileNormalization
from model_analysis.detector import Detector
from model_analysis.postprocessing.morphological_postprocessing import MorphologicalClosing


def load_fits_monochrome(image_path: Path) -> np.ndarray:
    with fits.open(image_path) as hdul:
        data = hdul[0].data
    if data is None:
        raise ValueError(f"Empty FITS image: {image_path}")
    if len(data.shape) == 3 and data.shape[0] == 3:
        rgb = np.rollaxis(data, 0, 3)
        return cv.cvtColor(rgb, cv.COLOR_BGR2GRAY).astype(np.float32)
    if len(data.shape) == 2:
        return data.astype(np.float32)
    raise ValueError(f"Unknown FITS shape: {data.shape}")

def parse_args():
    parser = argparse.ArgumentParser(
        prog="run_inference.py",
        description=(
            "Run sky object detection inference on a single FITS image.\n"
            "Normalizes the image, runs the model, applies postprocessing,\n"
            "and saves a visual result with the predicted mask and object positions."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python run_inference.py --model ./trained-models/model-2024.keras --image ./my_image.fits\n"
            "  python run_inference.py --model ./trained-models/model-2024.keras --image ./my_image.fits --output ./result.png --shape 512\n"
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
        "--image",
        required=True,
        type=Path,
        metavar="PATH",
        help="Path to the input FITS image file.",
    )
    parser.add_argument(
        "--output",
        default=Path("./infer_result.png"),
        type=Path,
        metavar="PATH",
        help="Path where the result plot will be saved. Default: ./infer_result.png",
    )
    parser.add_argument(
        "--shape",
        default=256,
        type=int,
        metavar="N",
        help="Target image size (square). Must match the size used during training. Default: 256",
    )
    return parser.parse_args()

def main():
    args = parse_args()

    target_shape   = (args.shape, args.shape)
    normalization  = LogPercentileNormalization()
    postprocessing = MorphologicalClosing()

    detect = Detector.from_saved_model(
        model_path=args.model,
        postprocessing=postprocessing,
        normalization=normalization,
        target_shape=target_shape,
    )

    raw_image = load_fits_monochrome(args.image)
    mask, positions = detect.predict_from_raw(raw_image)
    print(f"Detected {len(positions)} objects")

    norm_image = normalization.normalize(raw_image)
    resized    = cv.resize(norm_image, (target_shape[1], target_shape[0]))

    _, axs = plt.subplots(1, 3, figsize=(15, 5))

    axs[0].set_title("Input image")
    axs[0].imshow(resized, cmap="rainbow", origin="upper", norm=PowerNorm(gamma=0.3))
    axs[0].axis("off")

    axs[1].set_title("Predicted mask")
    axs[1].imshow(mask, cmap="viridis", origin="upper")
    axs[1].axis("off")

    axs[2].set_title(f"Detected objects ({len(positions)})")
    axs[2].imshow(resized, cmap="rainbow", origin="upper", norm=PowerNorm(gamma=0.3))
    if positions:
        xs, ys = zip(*positions)
        axs[2].scatter(xs, ys, s=15, edgecolors="black", facecolors="none")
    axs[2].axis("off")

    plt.tight_layout()
    plt.savefig(str(args.output), dpi=300)
    print(f"Result saved to: {args.output}")
    plt.show()


if __name__ == "__main__":
    main()