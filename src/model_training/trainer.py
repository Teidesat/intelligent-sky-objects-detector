from datetime import datetime
from pathlib import Path

from tensorflow import keras

from .modelling_specs.models.models_interface import ModelStrategy
from .modelling_specs.losses.losses_interface import LossStrategy


class Trainer:
    """
    Model construction, compilation, and training loop.
    """

    NUM_CLASSES = 2

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
        self.input_shape = input_shape
        self.output_dir = Path(output_dir)
        self.batch_size = batch_size
        self.epochs = epochs
        self.model: keras.Model | None = None

    def build(self) -> keras.Model:
        self.model = self.model_strategy.build(self.input_shape, self.NUM_CLASSES)
        self.model.compile(
            optimizer="adam",
            loss=self.loss_strategy.get_loss(),
            metrics=[keras.metrics.SparseCategoricalAccuracy()],
        )
        self.model.summary()
        return self.model

    def train(self, train_images, train_masks, val_images, val_masks) -> keras.callbacks.History:
        if self.model is None:
            raise RuntimeError("Call build() before train()")

        self.output_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_path = self.output_dir / f"model-{self._timestamp()}.ckpnt.keras"

        return self.model.fit(
            train_images,
            train_masks,
            validation_data=(val_images, val_masks),
            batch_size=self.batch_size,
            epochs=self.epochs,
            callbacks=[
                keras.callbacks.ModelCheckpoint(
                    filepath=str(checkpoint_path),
                    monitor="val_loss",
                    save_best_only=True,
                ),
            ],
        )

    def save(self) -> Path:
        if self.model is None:
            raise RuntimeError("No model to save")
        path = self.output_dir / f"model-{self._timestamp()}.keras"
        self.model.save(str(path))
        print(f"Model saved to: {path}")
        return path

    @staticmethod
    def _timestamp() -> str:
        return datetime.now().strftime("%Y_%m_%d-%H_%M_%S")