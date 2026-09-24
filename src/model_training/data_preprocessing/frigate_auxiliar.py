"""
================================================================================
⚠️  AVISO — PIPELINE DE FRIGATE EN ESTADO EXPERIMENTAL / NO RECOMENDADO
================================================================================

Este módulo forma parte del pipeline de preprocesado de Frigate y actualmente
NO funciona de forma fiable. Se mantiene en el repositorio como punto de
partida, pero requiere más investigación antes de poder usarse para entrenar
o evaluar modelos con garantías.

Problema conocido:
    La generación de máscaras (frame-differencing + threshold) no está
    respondiendo realmente al contenido de la imagen, sino a una
    codificación que se mantiene prácticamente constante entre frames.
    Es decir, el offset/escala usado al normalizar los datos crudos
    (actualmente `(data - 32768) / 32768`) no refleja el rango dinámico
    real de cada frame, por lo que la diferencia entre frames vecinos
    capta en buena medida un patrón de codificación fijo en lugar de
    variaciones reales de la escena (estrellas, streaks de satélites, etc.).
    Esto contamina las máscaras generadas y, por extensión, cualquier
    modelo entrenado con ellas.

Recomendación:
    NO usar este pipeline para entrenar o evaluar modelos por el momento.
    Antes de retomarlo, se recomienda:
      1. Inspeccionar visualmente el rango real de valores de varios FITS
         de Frigate (min/max/histograma) en lugar de asumir un offset fijo
         de 16 bits con signo.
      2. Verificar si el offset/escala de normalización debe calcularse
         por frame (o por sensor/sesión) en vez de usar una constante
         global.
      3. Confirmar, con inspección visual de las máscaras resultantes,
         que el patrón detectado corresponde a objetos reales en
         movimiento y no a artefactos de codificación.

El pipeline de Astrometry.net no se ve afectado por este problema y sigue
siendo la vía principal y fiable del proyecto.
================================================================================
"""

import numpy as np
import cv2
import astropy.io.fits as fits
from pathlib import Path
from tqdm import tqdm
from collections import deque


def _compute_raw_diff_mask(buffer: deque, diff_window: int, threshold_sigma: float) -> np.ndarray:
    """
    Computes the thresholded, morphologically-cleaned difference mask for the
    frame currently at the centre of the buffer, at native resolution and
    BEFORE any deliberate enlargement (downscale-robustness dilation, final
    radius dilation). Factored out so both the artifact-detection pass and
    the final pair-generation pass use the exact same computation.
    """
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
    return mask_raw


def detect_static_artifacts(
    input_dir: Path,
    max_frames: int = 200,
    diff_window: int = 10,
    threshold_sigma: float = 1.5,
    artifact_fraction: float = 0.05,
    artifact_dilate: int = 2,
) -> np.ndarray | None:
    """
    FIRST PASS — detects sensor-level static artifacts (hot/dead/stuck pixels).

    Without dark-frame calibration (Frigate's own documentation notes GAIN and
    CCD-TEMP were left unset, and no dark/flat frames were provided), a hot
    pixel that flickers in value frame-to-frame — while never actually moving —
    registers as a "moving object" in the frame-differencing mask on nearly
    every frame. A real satellite/debris streak, in contrast, only transits
    a handful of consecutive frames before leaving the field of view.

    This pass accumulates, at native FITS resolution, how many times each
    pixel is flagged positive across the whole sequence. Any pixel flagged
    in more than `artifact_fraction` of all processed frames is treated as a
    static sensor defect rather than a real transient, and is returned as a
    boolean artifact map to be excluded from the final training masks.

    Returns:
        artifact_mask: bool array (H, W) at native FITS resolution, True where
                       a pixel is a suspected static artifact. None if no
                       frames could be processed.
    """
    fits_files = sorted(input_dir.glob("*.fits"))[:max_frames]
    print(f"[Pass 1/2] Scanning {len(fits_files)} files for static sensor artifacts (hot/dead pixels)...")

    buffer = deque(maxlen=2 * diff_window + 1)
    hit_counter = None
    frames_processed = 0

    for f in tqdm(fits_files, desc="Detecting artifacts"):
        with fits.open(f) as hdul:
            data = hdul[0].data.astype(np.float32)
            img = (data - 32768) / 32768.0
            img = np.clip(img, 0, 1)
            buffer.append(img)

        if len(buffer) < 2 * diff_window + 1:
            continue

        mask_raw = _compute_raw_diff_mask(buffer, diff_window, threshold_sigma)

        if hit_counter is None:
            hit_counter = np.zeros(mask_raw.shape, dtype=np.int32)

        hit_counter += (mask_raw > 0).astype(np.int32)
        frames_processed += 1

    if hit_counter is None or frames_processed == 0:
        print("[Pass 1/2] No frames processed — skipping artifact detection.")
        return None

    hit_fraction = hit_counter.astype(np.float32) / frames_processed
    artifact_mask = hit_fraction > artifact_fraction

    n_artifact_px = int(artifact_mask.sum())
    print(f"[Pass 1/2] Frames analysed: {frames_processed}")
    print(f"[Pass 1/2] Static artifact pixels found: {n_artifact_px} "
          f"(flagged positive in >{artifact_fraction:.0%} of frames — "
          f"almost certainly hot/dead pixels, not real transiting objects)")

    if n_artifact_px > 0:
        # Slightly dilate the artifact map: a hot pixel's morphological footprint
        # after MORPH_OPEN + dilate(3x3) can bleed a pixel or two into its
        # neighbours, so we suppress a small margin around each flagged pixel too.
        dilate_kernel = np.ones((2 * artifact_dilate + 1, 2 * artifact_dilate + 1), np.uint8)
        artifact_mask = cv2.dilate(artifact_mask.astype(np.uint8), dilate_kernel, iterations=1).astype(bool)

    return artifact_mask


def generate_pairs(
    input_dir: Path,
    output_dir: Path,
    target_size: tuple = (256, 256),
    max_frames: int = 200,
    diff_window: int = 10,
    threshold_sigma: float = 1.5,
    final_mask_radius: int = 5,
    artifact_fraction: float = 0.05,
    skip_artifact_detection: bool = False,
):
    """
    Generates pairs of images and masks from FITS files in the input directory. 
    The masks are created by calculating the difference between the average of the previous and next frames, 
    applying a threshold, and performing morphological operations to clean up the mask. 
    The resulting images and masks are resized to the target size and saved as .npy files in the output directory.

    Runs a first pass (see `detect_static_artifacts`) to identify sensor-level
    static defects (hot/dead pixels) that would otherwise be marked as
    "moving objects" in every single frame, teaching the model to always fire
    at the same fixed image coordinate regardless of actual content.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    fits_files = sorted(input_dir.glob("*.fits"))[:max_frames]
    print(f"Procesando {len(fits_files)} archivos...")

    # ------------------------------------------------------------------
    # PASS 1 — detect static sensor artifacts (hot/dead pixels) so they can
    # be excluded from every mask generated in Pass 2. See docstring above
    # and detect_static_artifacts() for the full rationale.
    # ------------------------------------------------------------------
    artifact_mask = None
    if not skip_artifact_detection:
        artifact_mask = detect_static_artifacts(
            input_dir=input_dir,
            max_frames=max_frames,
            diff_window=diff_window,
            threshold_sigma=threshold_sigma,
            artifact_fraction=artifact_fraction,
        )
    else:
        print("Artifact detection skipped (--skip-artifact-detection).")

    # Deque to hold the last (2*diff_window + 1) images for difference calculation
    buffer = deque(maxlen=2 * diff_window + 1)

    total_saved = 0
    total_mask_pixels = 0
    empty_masks = 0
    total_artifact_px_suppressed = 0

    print(f"[Pass 2/2] Generating image/mask pairs...")
    for i, f in enumerate(tqdm(fits_files, desc="Generating pairs")):
        with fits.open(f) as hdul:
            data = hdul[0].data.astype(np.float32)
            img = (data - 32768) / 32768.0
            img = np.clip(img, 0, 1)
            buffer.append(img)

        if len(buffer) < 2 * diff_window + 1:
            continue

        current_img = buffer[diff_window]

        mask_raw = _compute_raw_diff_mask(buffer, diff_window, threshold_sigma)

        # NEW: suppress known static sensor artifacts (hot/dead pixels) BEFORE
        # any further enlargement, so they never enter the training mask,
        # regardless of how consistently they trip the diff threshold.
        if artifact_mask is not None:
            n_suppressed = int(((mask_raw > 0) & artifact_mask).sum())
            total_artifact_px_suppressed += n_suppressed
            mask_raw[artifact_mask] = 0

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
        if artifact_mask is not None:
            print(f"   Píxeles de artefacto suprimidos (total, todos los frames): "
                  f"{total_artifact_px_suppressed}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--size", type=int, default=256)
    parser.add_argument("--max", type=int, default=200)
    parser.add_argument("--mask-radius", type=int, default=5,
                        help="Radio final del objeto en espacio 256x256 (default: 5)")
    parser.add_argument("--artifact-fraction", type=float, default=0.05,
                        help="Fracción de frames en la que un píxel debe aparecer marcado "
                             "como positivo para considerarse un artefacto estático del sensor "
                             "(hot/dead pixel) en vez de un objeto real en tránsito. Default: 0.05 (5%%)")
    parser.add_argument("--skip-artifact-detection", action="store_true",
                        help="Desactiva la primera pasada de detección de artefactos estáticos "
                             "(hot/dead pixels). No recomendado salvo para depuración rápida.")
    args = parser.parse_args()
    generate_pairs(args.input, args.output, (args.size, args.size), args.max,
                   final_mask_radius=args.mask_radius,
                   artifact_fraction=args.artifact_fraction,
                   skip_artifact_detection=args.skip_artifact_detection)