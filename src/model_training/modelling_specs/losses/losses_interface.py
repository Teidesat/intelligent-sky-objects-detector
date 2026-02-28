from abc import ABC, abstractmethod
from typing import Callable


class LossStrategy(ABC):
    """Base class for all loss function strategies."""

    @abstractmethod
    def get_loss(self) -> Callable:
        ...
