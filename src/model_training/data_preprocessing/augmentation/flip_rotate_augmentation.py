from .augmentation_interface import AugmentationStrategy

import random
import torch


class FlipRotateAugmentation(AugmentationStrategy):

    def __init__(self,
                 use_flips: bool = True,
                 use_rotations: bool = True):
        self.use_flips = use_flips
        self.use_rotations = use_rotations

    def __call__(self, image, mask):

        if self.use_flips:

            if random.random() < 0.5:
                image = torch.flip(image, dims=[2])
                mask  = torch.flip(mask, dims=[1])

            if random.random() < 0.5:
                image = torch.flip(image, dims=[1])
                mask  = torch.flip(mask, dims=[0])

        if self.use_rotations:
            k = random.randint(0, 3)

            image = torch.rot90(image, k, dims=[1, 2])
            mask  = torch.rot90(mask, k, dims=[0, 1])

        return image, mask