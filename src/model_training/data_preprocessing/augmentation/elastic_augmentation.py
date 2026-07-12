import random
import torch
import torch.nn.functional as F
from .augmentation_interface import AugmentationStrategy

class ElasticAugmentation(AugmentationStrategy):
    def __init__(self, alpha: float = 10.0, sigma: float = 4.0, prob: float = 0.3):
        self.alpha = alpha
        self.sigma = sigma
        self.prob = prob

    def __call__(self, image: torch.Tensor, mask: torch.Tensor):
        if random.random() > self.prob:
            return image, mask

        _, h, w = image.shape
        device = image.device

        # 1. Generar campos de desplazamiento aleatorios suaves
        dx = torch.randn(1, 1, h, w, device=device) * self.alpha
        dy = torch.randn(1, 1, h, w, device=device) * self.alpha

        # Suavizado con convolución (kernel promedio)
        kernel_size = int(self.sigma * 2) + 1
        if kernel_size % 2 == 0:
            kernel_size += 1
        kernel = torch.ones(1, 1, kernel_size, kernel_size, device=device) / (kernel_size * kernel_size)
        dx = F.conv2d(dx, kernel, padding=kernel_size//2)
        dy = F.conv2d(dy, kernel, padding=kernel_size//2)

        # Eliminar dimensiones extra -> (H, W)
        dx = dx.squeeze(0).squeeze(0)
        dy = dy.squeeze(0).squeeze(0)

        # 2. Crear malla de coordenadas normalizadas [-1, 1]
        y_grid, x_grid = torch.meshgrid(
            torch.linspace(-1.0, 1.0, h, device=device),
            torch.linspace(-1.0, 1.0, w, device=device),
            indexing='ij'
        )
        # Aplicar desplazamiento escalado a rango [-1,1]
        norm_h = 2.0 / (h - 1.0)
        norm_w = 2.0 / (w - 1.0)
        grid_x = x_grid + dx * norm_w
        grid_y = y_grid + dy * norm_h

        # 3. Crear grid (1, H, W, 2) – sin unsqueeze extra
        grid = torch.stack([grid_x, grid_y], dim=2).unsqueeze(0)

        # 4. Aplicar warp a imagen y máscara
        img_batch = image.unsqueeze(0)  # (1, C, H, W)
        mask_batch = mask.unsqueeze(0).unsqueeze(0).float()  # (1, 1, H, W)

        warped_img = F.grid_sample(img_batch, grid, mode='bilinear', padding_mode='border', align_corners=False)
        warped_mask = F.grid_sample(mask_batch, grid, mode='bilinear', padding_mode='border', align_corners=False)

        # Volver a forma original
        warped_img = warped_img.squeeze(0)
        warped_mask = warped_mask.squeeze(0).squeeze(0)
        warped_mask = (warped_mask > 0.5).float()

        return warped_img, warped_mask