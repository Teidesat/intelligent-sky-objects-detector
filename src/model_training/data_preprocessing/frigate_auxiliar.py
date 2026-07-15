import numpy as np
import cv2
import astropy.io.fits as fits
from pathlib import Path
from tqdm import tqdm
from collections import deque

def generate_pairs(
    input_dir: Path,
    output_dir: Path,
    target_size: tuple = (256, 256),
    max_frames: int = 200,
    diff_window: int = 10,
    threshold_sigma: float = 1.5,
    final_mask_radius: int = 5,
):
    """
    Generates pairs of images and masks from FITS files in the input directory. 
    The masks are created by calculating the difference between the average of the previous and next frames, 
    applying a threshold, and performing morphological operations to clean up the mask. 
    The resulting images and masks are resized to the target size and saved as .npy files in the output directory.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    fits_files = sorted(input_dir.glob("*.fits"))[:max_frames]
    print(f"Procesando {len(fits_files)} archivos...")

    # Deque to hold the last (2*diff_window + 1) images for difference calculation
    buffer = deque(maxlen=2 * diff_window + 1)

    total_saved = 0
    total_mask_pixels = 0
    empty_masks = 0

    for i, f in enumerate(tqdm(fits_files)):
        with fits.open(f) as hdul:
            data = hdul[0].data.astype(np.float32)
            img = (data - 32768) / 32768.0
            img = np.clip(img, 0, 1)
            buffer.append(img)

        if len(buffer) < 2 * diff_window + 1:
            continue

        current_img = buffer[diff_window]

        prev_avg = np.mean(list(buffer)[:diff_window], axis=0)
        next_avg = np.mean(list(buffer)[diff_window+1:], axis=0)
        diff = np.abs(prev_avg - next_avg)

        mean_diff = np.mean(diff)
        std_diff = np.std(diff)
        thresh = mean_diff + threshold_sigma * std_diff
        mask_raw = (diff > thresh).astype(np.uint8) * 255

        kernel = np.ones((3, 3), np.uint8)
        mask_raw = cv2.morphologyEx(mask_raw, cv2.MORPH_OPEN, kernel)
        mask_raw = cv2.dilate(mask_raw, kernel, iterations=1)

        # Downscale of ~9600x6422 -> 256x256 is a factor of ~37x per axis.
        # A thin streak (2-3 px) rarely survives a resize with INTER_NEAREST,
        # because it simply samples a point every ~37 px and may fall outside the streak.
        scale_x = current_img.shape[1] / target_size[0]
        scale_y = current_img.shape[0] / target_size[1]
        downscale_factor = int(round(max(scale_x, scale_y)))
        prep_kernel_size = max(3, downscale_factor) 
        prep_kernel = np.ones((prep_kernel_size, prep_kernel_size), np.uint8)
        mask_raw = cv2.dilate(mask_raw, prep_kernel, iterations=1)

        img_resized = cv2.resize(current_img, target_size, interpolation=cv2.INTER_AREA)

        mask_float = mask_raw.astype(np.float32) / 255.0
        mask_resized_float = cv2.resize(mask_float, target_size, interpolation=cv2.INTER_AREA)
        mask_resized = (mask_resized_float > 0.0).astype(np.uint8)

        final_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                                  (2 * final_mask_radius + 1, 2 * final_mask_radius + 1))
        mask_resized = cv2.dilate(mask_resized, final_kernel, iterations=1)

        mask_pixels = int(mask_resized.sum())
        total_mask_pixels += mask_pixels
        if mask_pixels == 0:
            empty_masks += 1

        entry_id = fits_files[i].stem
        np.save(output_dir / f"{entry_id}_image.npy", img_resized)
        np.save(output_dir / f"{entry_id}_mask.npy", mask_resized)
        total_saved += 1


    print(f"Guardados {max(0, len(fits_files) - 2*diff_window)} pares en {output_dir}")
    if total_saved > 0:
        print(f"   Máscaras vacías: {empty_masks}/{total_saved} "
              f"({100*empty_masks/total_saved:.1f}%)")
        print(f"   Media de píxeles positivos por máscara: {total_mask_pixels/total_saved:.1f}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--size", type=int, default=256)
    parser.add_argument("--max", type=int, default=200)
    parser.add_argument("--mask-radius", type=int, default=5,
                        help="Radio final del objeto en espacio 256x256 (default: 5)")
    args = parser.parse_args()
    generate_pairs(args.input, args.output, (args.size, args.size), args.max,
                   final_mask_radius=args.mask_radius)