from datetime import datetime
from pathlib import Path
from typing import Callable

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from .modelling_specs.models.models_interface import ModelStrategy
from .modelling_specs.losses.losses_interface import LossStrategy


class History:
    """Equivalent to Keras History — stores per-epoch metrics."""

    def __init__(self):
        self.history: dict[str, list[float]] = {
            "loss": [], "acc": [], "iou": [],
            "val_loss": [], "val_acc": [], "val_iou": [],
        }


class Trainer:
    """Model construction, compilation, and training loop."""

    NUM_CLASSES = 1

    def __init__(
        self,
        model_strategy: ModelStrategy,
        loss_strategy: LossStrategy,
        input_shape: tuple,
        output_dir: Path,
        batch_size: int = 6,
        epochs: int = 50,
    ):
        self.model_strategy = model_strategy
        self.loss_strategy = loss_strategy
        self.input_shape = input_shape          # (H, W, C)
        self.output_dir = Path(output_dir)
        self.batch_size = batch_size
        self.epochs = epochs
        self.model: nn.Module | None = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def build(self) -> nn.Module:
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = True      # False
        torch.backends.cudnn.deterministic = False   # True

        self.model = self.model_strategy.build(self.input_shape, self.NUM_CLASSES)
        self.model.to(self.device)
        print(self.model)
        return self.model

    def train(
        self,
        train_images: torch.Tensor,
        train_masks: torch.Tensor,
        val_images: torch.Tensor,
        val_masks: torch.Tensor,
    ) -> History:
        if self.model is None:
            raise RuntimeError("Call build() before train()")

        self.output_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_path = self.output_dir / f"model-{self._timestamp()}.ckpt.pt"

        train_loader = self._make_loader(train_images, train_masks, shuffle=True)
        val_loader   = self._make_loader(val_images,   val_masks,   shuffle=False)

        criterion = self.loss_strategy.get_loss()
        if isinstance(criterion, nn.Module):
            criterion = criterion.to(self.device)

        optimizer = optim.Adam(self.model.parameters())

        history = History()
        best_val_loss = float("inf")

        for epoch in range(1, self.epochs + 1):
            train_metrics = self._run_epoch(train_loader, criterion, optimizer, training=True, epoch=epoch)
            val_metrics   = self._run_epoch(val_loader,  criterion, optimizer=None, training=False, epoch=epoch)

            self._log_epoch(epoch, train_metrics, val_metrics)
            self._update_history(history, train_metrics, val_metrics)

            if val_metrics["loss"] < best_val_loss:
                best_val_loss = val_metrics["loss"]
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                torch.save(self.model.state_dict(), checkpoint_path)
                print(f"  ✓ Checkpoint saved ({checkpoint_path.name})")

        return history

    def save(self) -> Path:
        if self.model is None:
            raise RuntimeError("No model to save")
        path = self.output_dir / f"model-{self._timestamp()}.pt"
        torch.save(self.model, str(path))
        print(f"Model saved to: {path}")
        return path

    def _make_loader(
        self,
        images: torch.Tensor,
        masks: torch.Tensor,
        shuffle: bool,
    ) -> DataLoader:
        # images: (B, H, W, 1) → (B, 1, H, W)
        images_chw = images.permute(0, 3, 1, 2)
        return DataLoader(
            TensorDataset(images_chw, masks),
            batch_size=self.batch_size,
            shuffle=shuffle,
        )

    def _run_epoch(
        self,
        loader: DataLoader,
        criterion: Callable,
        optimizer: optim.Optimizer | None,
        training: bool,
        epoch = None,
    ) -> dict[str, float]:
        self.model.train(training)
        total_loss = total_acc = total_iou = 0.0
        n = len(loader)

        with (torch.enable_grad() if training else torch.no_grad()):
            for imgs, masks in loader:
                imgs, masks = imgs.to(self.device), masks.to(self.device)

                if training:
                    optimizer.zero_grad()

                preds = self.model(imgs)                        # (B, C, H, W)
                loss  = criterion(preds.squeeze(1), masks.float())

                if training:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    optimizer.step()

                acc, iou = self._compute_metrics(preds.detach(), masks)
                total_loss += loss.item()
                total_acc  += acc
                total_iou  += iou

        return {"loss": total_loss / n, "acc": total_acc / n, "iou": total_iou / n}

    @staticmethod
    def _compute_metrics(
        y_pred: torch.Tensor,
        y_true: torch.Tensor,
    ) -> tuple[float, float]:
        """Binary accuracy and IoU from sigmoid single-channel output."""
        pred_bin = (y_pred[:, 0] > 0.5).float()
        true     = y_true.float()

        acc = (pred_bin == true).float().mean().item()

        intersection = (pred_bin * true).sum().item()
        union        = (pred_bin + true).clamp(max=1).sum().item()
        iou          = (intersection + 1e-6) / (union + 1e-6)

        return acc, iou

    def _log_epoch(
        self,
        epoch: int,
        train: dict[str, float],
        val: dict[str, float],
    ) -> None:
        print(
            f"Epoch {epoch:>3}/{self.epochs} — "
            f"loss: {train['loss']:.4f}  acc: {train['acc']:.4f}  iou: {train['iou']:.4f} | "
            f"val_loss: {val['loss']:.4f}  val_acc: {val['acc']:.4f}  val_iou: {val['iou']:.4f}"
        )

    @staticmethod
    def _update_history(
        history: History,
        train: dict[str, float],
        val: dict[str, float],
    ) -> None:
        history.history["loss"].append(train["loss"])
        history.history["acc"].append(train["acc"])
        history.history["iou"].append(train["iou"])
        history.history["val_loss"].append(val["loss"])
        history.history["val_acc"].append(val["acc"])
        history.history["val_iou"].append(val["iou"])

    @staticmethod
    def _timestamp() -> str:
        return datetime.now().strftime("%Y_%m_%d-%H_%M_%S")