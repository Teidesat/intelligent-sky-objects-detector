import time
from datetime import datetime
from pathlib import Path
import json

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from .modelling_specs.models.models_interface import ModelStrategy
from .modelling_specs.losses.losses_interface import LossStrategy
from .data_preprocessing.augmentation.augmentation_interface import AugmentationStrategy
from .data_preprocessing.augmentation.segmentation_dataset import SegmentationDataset


class History:
    """Equivalent to Keras History — stores per-epoch metrics."""

    def __init__(self):
        self.history: dict[str, list[float]] = {
            "loss": [], "acc": [], "iou": [], "precision": [], "recall": [],
            "specificity": [], "f1": [],
            "val_loss": [], "val_acc": [], "val_iou": [], "val_precision": [],
            "val_recall": [], "val_specificity": [], "val_f1": [],
        }


class Trainer:
    """Model construction, compilation, and training loop."""

    def __init__(
        self,
        model_strategy: ModelStrategy,
        loss_strategy: LossStrategy,
        input_shape: tuple,
        output_dir: Path,
        augmentation_strategy: AugmentationStrategy | None = None,
        batch_size: int = 6,
        epochs: int = 100,
    ):
        self.model_strategy = model_strategy
        self.loss_strategy = loss_strategy
        self.input_shape = input_shape          # (H, W, C)
        self.output_dir = Path(output_dir)
        self.augmentation_strategy = augmentation_strategy
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
        torch.backends.cudnn.benchmark = True     
        torch.backends.cudnn.deterministic = False   

        self.model = self.model_strategy.build(
            self.input_shape, 
            self.loss_strategy.output_channels,
            apply_activation=self.loss_strategy.needs_activation,
        )
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

        train_loader = self._make_loader(train_images, train_masks, shuffle=True, training=True)
        val_loader   = self._make_loader(val_images,   val_masks,   shuffle=False)

        criterion = self.loss_strategy.get_loss()
        if isinstance(criterion, nn.Module):
            criterion = criterion.to(self.device)
        # 1e-4 para lovasz y tversky, 1e-3 para BCE y Dice. 1e-3 es el default de Adam.
        optimizer = optim.Adam(self.model.parameters(), lr=1e-3, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', factor=0.5, patience=10, min_lr=1e-6 # 1e-6
        )

        history = History()
        best_val_iou = 0.0

        for epoch in range(1, self.epochs + 1):
            epoch_start_time = time.time()

            train_metrics = self._run_epoch(train_loader, criterion, optimizer, training=True, epoch=epoch)
            val_metrics   = self._run_epoch(val_loader,  criterion, optimizer=None, training=False, epoch=epoch)
            
            epoch_end_time = time.time()
            epoch_duration = epoch_end_time - epoch_start_time
    
            # Formatear el tiempo en minutos y segundos
            minutes = int(epoch_duration // 60)
            seconds = epoch_duration % 60
            if minutes > 0:
                epoch_time_str = f"{minutes}m {seconds:.2f}s"
            else:
                epoch_time_str = f"{seconds:.2f}s"

            self._log_epoch(epoch, train_metrics, val_metrics, epoch_time=epoch_time_str)
            self._update_history(history, train_metrics, val_metrics)

            if val_metrics["iou"] > best_val_iou:  
                best_val_iou = val_metrics["iou"]
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                torch.save(self.model.state_dict(), checkpoint_path)
                print(f"  ✓ Checkpoint saved ({checkpoint_path.name})")

            scheduler.step(val_metrics["iou"])
            print(f"  LR: {optimizer.param_groups[0]['lr']:.2e}")

        config = {
            "model": self.model_strategy.__class__.__name__,
            "loss": self.loss_strategy.__class__.__name__,
            "loss_params": self.loss_strategy.__dict__,
            "augmentation": self.augmentation_strategy.__class__.__name__ if self.augmentation_strategy else None,
            "batch_size": self.batch_size,
            "epochs": self.epochs,
            "input_shape": self.input_shape,
            "device": str(self.device),
            "timestamp": datetime.now().isoformat(),
        }
        self._save_experiment_logs(history, config)

        return history

    def save(self) -> Path:
        if self.model is None:
            raise RuntimeError("No model to save")
        path = self.output_dir / f"model-{self._timestamp()}.pt"
        checkpoint = {
            "model": self.model,
            "output_channels": self.loss_strategy.output_channels,
            "needs_activation": self.loss_strategy.needs_activation,
        }
        torch.save(checkpoint, str(path))
        print(f"Model saved to: {path}")
        return path

    def _make_loader(self, images, masks, shuffle: bool, training=False) -> DataLoader:
        dataset = SegmentationDataset(
            images,
            masks,
            augmentation=self.augmentation_strategy if training else None
        )

        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            pin_memory=True,
        )

    def _run_epoch(self, loader, criterion, optimizer, training, epoch=None) -> dict[str, float]:
        self.model.train(training)
        total_loss = total_acc = total_iou = 0.0
        total_precision = total_recall = total_specificity = total_f1 = 0.0
        n = len(loader)

        with (torch.enable_grad() if training else torch.no_grad()):
            for imgs, masks in loader:
                imgs, masks = imgs.to(self.device), masks.to(self.device)

                if training:
                    optimizer.zero_grad()

                preds = self.model(imgs)  # (B, C, H, W) — C según loss_strategy
                preds_for_loss   = self.loss_strategy.format_predictions(preds)
                targets_for_loss = self.loss_strategy.format_targets(masks)
                loss = criterion(preds_for_loss, targets_for_loss)

                if training:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    optimizer.step()

                probs = self.loss_strategy.predictions_to_probability(preds.detach())  # (B,H,W) en [0,1]
                metrics = self._compute_metrics(probs, masks)
                total_loss        += loss.item()
                total_acc         += metrics["acc"]
                total_iou         += metrics["iou"]
                total_precision   += metrics["precision"]
                total_recall      += metrics["recall"]
                total_specificity += metrics["specificity"]
                total_f1          += metrics["f1"]

        return {
            "loss": total_loss / n, "acc": total_acc / n, "iou": total_iou / n,
            "precision": total_precision / n, "recall": total_recall / n,
            "specificity": total_specificity / n, "f1": total_f1 / n,
        }

    @staticmethod
    def _compute_metrics(
        y_pred: torch.Tensor,
        y_true: torch.Tensor,
    ) -> dict[str, float]:
        """
        Binary segmentation metrics from sigmoid single-channel output.
        - accuracy:    (TP + TN) / total
        - precision:   TP / (TP + FP)  — de lo que predice estrella, cuánto acierta
        - recall:      TP / (TP + FN)  — sensibilidad: de las estrellas reales, cuántas detecta
        - specificity: TN / (TN + FP)  — de los fondos reales, cuántos clasifica bien
        - f1:          media armónica de precision y recall
        - iou:         TP / (TP + FP + FN)
        """
        pred_bin = (y_pred > 0.3).float()
        true     = y_true.float()

        tp = (pred_bin * true).sum().item()
        tn = ((1 - pred_bin) * (1 - true)).sum().item()
        fp = (pred_bin * (1 - true)).sum().item()
        fn = ((1 - pred_bin) * true).sum().item()

        acc         = (tp + tn) / (tp + tn + fp + fn + 1e-6)
        precision   = (tp + 1e-6) / (tp + fp + 1e-6)
        recall      = (tp + 1e-6) / (tp + fn + 1e-6)
        specificity = (tn + 1e-6) / (tn + fp + 1e-6)
        f1          = 2 * (precision * recall) / (precision + recall + 1e-6)
        iou         = (tp + 1e-6) / (tp + fp + fn + 1e-6)

        return {
            "acc": acc, "iou": iou,
            "precision": precision, "recall": recall,
            "specificity": specificity, "f1": f1,
        }

    def _log_epoch(
        self,
        epoch: int,
        train: dict[str, float],
        val: dict[str, float],
        epoch_time: str,
    ) -> None:
        print(
            f"Epoch {epoch:>3}/{self.epochs} [{epoch_time}]\n"
            f"  TRAIN  loss: {train['loss']:.4f}  iou: {train['iou']:.4f}  f1: {train['f1']:.4f}"
            f"  prec: {train['precision']:.4f}  rec: {train['recall']:.4f}  spec: {train['specificity']:.4f}\n"
            f"  VAL    loss: {val['loss']:.4f}  iou: {val['iou']:.4f}  f1: {val['f1']:.4f}"
            f"  prec: {val['precision']:.4f}  rec: {val['recall']:.4f}  spec: {val['specificity']:.4f}"
        )

    @staticmethod
    def _update_history(
        history: History,
        train: dict[str, float],
        val: dict[str, float],
    ) -> None:
        for key in ["loss", "acc", "iou", "precision", "recall", "specificity", "f1"]:
            history.history[key].append(train[key])
            history.history[f"val_{key}"].append(val[key])

    @staticmethod
    def _timestamp() -> str:
        return datetime.now().strftime("%Y_%m_%d-%H_%M_%S")
    
    def _save_experiment_logs(self, history: History, config: dict):
        """Guarda el historial de métricas y la configuración en JSON."""
        # Crear carpeta de logs si no existe
        log_dir = self.output_dir / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        
        # Usar el timestamp del checkpoint para nombrar los archivos
        timestamp = datetime.now().strftime("%Y_%m_%d-%H_%M_%S")
        
        # Guardar historial (métricas por época)
        history_path = log_dir / f"history_{timestamp}.json"
        with open(history_path, "w") as f:
            json.dump(history.history, f, indent=2)
        print(f"  ✓ History saved to: {history_path}")
        
        # Guardar configuración del experimento
        config_path = log_dir / f"config_{timestamp}.json"
        with open(config_path, "w") as f:
            json.dump(config, f, indent=2, default=str)  # default=str para manejar objetos no serializables
        print(f"  ✓ Config saved to: {config_path}")