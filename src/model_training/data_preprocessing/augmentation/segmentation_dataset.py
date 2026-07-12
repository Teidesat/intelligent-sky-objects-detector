from torch.utils.data import Dataset

class SegmentationDataset(Dataset):
    def __init__(self, images, masks, augmentation=None):
        self.images = images
        self.masks = masks
        self.augmentation = augmentation

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]      # HWC
        mask = self.masks[idx]        # HW

        image = image.permute(2, 0, 1)  # CHW

        if self.augmentation:
            image, mask = self.augmentation(image, mask)

        return image, mask