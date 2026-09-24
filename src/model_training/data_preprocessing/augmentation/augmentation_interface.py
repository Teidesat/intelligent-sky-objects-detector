from abc import ABC, abstractmethod
import torch


class AugmentationStrategy(ABC):

    @abstractmethod
    def __call__(
        self,
        image: torch.Tensor,
        mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        pass