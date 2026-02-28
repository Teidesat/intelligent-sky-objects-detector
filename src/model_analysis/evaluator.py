import random
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import PowerNorm

from .detector import Detector


class Evaluator:
    """Evaluates a trained model on a dataset and produces visual reports."""

    def __init__(self, detector: Detector, output_dir: Path):
        self.detector = detector
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def plot_loss_curves(self, history, filename: str = "loss_history.png") -> Path:
        """Plot training vs validation loss curves from a Keras History object."""
        fig, ax = plt.subplots(1, 1, figsize=(10, 5))
        ax.plot(history.history["loss"], label="Train loss")
        ax.plot(history.history["val_loss"], label="Validation loss")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.legend()
        save_path = self.output_dir / filename
        plt.savefig(str(save_path))
        plt.show()
        print(f"Loss curves saved to: {save_path}")
        return save_path

    def plot_dataset_sample(
        self,
        dataset: dict,
        n_samples: int = 27,
        rows: int = 9,
        cols: int = 3,
        filename: str = "dataset_sample.png",
    ) -> Path:
        """Plot a grid of (input image | objects scatter | ground truth mask)."""
        entries = random.sample(list(dataset.values()), min(n_samples, len(dataset)))
        _, axs = plt.subplots(rows, cols * 3, figsize=(cols * 9, rows * 3))

        for i in range(rows):
            for j in range(cols):
                idx = i * cols + j
                if idx >= len(entries):
                    break
                entry = entries[idx]
                ax_img  = axs[i, j * 3]
                ax_obj  = axs[i, j * 3 + 1]
                ax_mask = axs[i, j * 3 + 2]

                ax_img.axis("off")
                ax_img.set_title(f"id: {entry.entry_id}")
                ax_img.imshow(entry.nn_input_image, cmap="rainbow", origin="upper", norm=PowerNorm(gamma=0.3))

                ax_obj.axis("off")
                ax_obj.set_title("detected objects")
                ax_obj.imshow(entry.nn_input_image, cmap="rainbow", origin="upper", norm=PowerNorm(gamma=0.3))
                if entry.filtered_objects:
                    xs, ys, _ = zip(*entry.filtered_objects)
                    ax_obj.scatter(xs, ys, s=10, edgecolors="black", facecolors="none")

                ax_mask.axis("off")
                ax_mask.set_title("segmentation mask")
                ax_mask.imshow(entry.segmentation_mask, cmap="viridis", origin="upper")

        plt.tight_layout()
        save_path = self.output_dir / filename
        plt.savefig(str(save_path), dpi=500)
        plt.show()
        print(f"Dataset sample saved to: {save_path}")
        return save_path

    def plot_inference_grid(
        self,
        test_dataset: dict,
        n_samples: int = 27,
        rows: int = 9,
        cols: int = 3,
        filename: str = "inference_result.png",
    ) -> Path:
        """Plot a grid of (input image | predicted mask | detected positions)."""
        entries = random.sample(list(test_dataset.values()), min(n_samples, len(test_dataset)))
        _, axs = plt.subplots(rows, cols * 3, figsize=(cols * 9, rows * 3))

        for i in range(rows):
            for j in range(cols):
                idx = i * cols + j
                if idx >= len(entries):
                    break
                entry = entries[idx]
                image = entry.nn_input_image

                mask = self.detector.predict_mask(image)
                positions = self.detector.postprocessing.extract_positions(mask)

                ax_img  = axs[i, j * 3]
                ax_mask = axs[i, j * 3 + 1]
                ax_pos  = axs[i, j * 3 + 2]

                ax_img.axis("off")
                ax_img.set_title(f"id: {entry.entry_id}")
                ax_img.imshow(image, cmap="rainbow", origin="upper", norm=PowerNorm(gamma=0.3))

                ax_mask.axis("off")
                ax_mask.set_title("predicted mask")
                ax_mask.imshow(mask, cmap="viridis", origin="upper")

                ax_pos.axis("off")
                ax_pos.set_title(f"positions ({len(positions)})")
                ax_pos.imshow(image, cmap="rainbow", origin="upper", norm=PowerNorm(gamma=0.3))
                if positions:
                    xs, ys = zip(*positions)
                    ax_pos.scatter(xs, ys, s=15, edgecolors="black", facecolors="none")

        plt.tight_layout()
        save_path = self.output_dir / filename
        plt.savefig(str(save_path), dpi=500)
        plt.show()
        print(f"Inference grid saved to: {save_path}")
        return save_path