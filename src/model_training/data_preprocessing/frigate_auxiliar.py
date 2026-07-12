# import numpy as np
# import cv2
# import astropy.io.fits as fits
# from pathlib import Path
# from tqdm import tqdm
# from collections import deque

# def generate_pairs(
#     input_dir: Path,
#     output_dir: Path,
#     target_size: tuple = (256, 256),
#     max_frames: int = 200,
#     diff_window: int = 10,
#     threshold_sigma: float = 1.5,
# ):
#     output_dir.mkdir(parents=True, exist_ok=True)
#     fits_files = sorted(input_dir.glob("*.fits"))[:max_frames]
#     print(f"Procesando {len(fits_files)} archivos...")

#     # Usaremos una cola (deque) para mantener solo las últimas imágenes necesarias
#     # En lugar de cargar todas, cargamos una ventana de tamaño (2*diff_window + 1)
#     # y vamos desechando las antiguas.
#     buffer = deque(maxlen=2 * diff_window + 1)  # guarda imágenes en float32

#     for i, f in enumerate(tqdm(fits_files)):
#         with fits.open(f) as hdul:
#             data = hdul[0].data.astype(np.float32)
#             img = (data - 32768) / 32768.0
#             img = np.clip(img, 0, 1)
#             buffer.append(img)

#         # Solo empezamos a procesar cuando el buffer está lleno
#         if len(buffer) < 2 * diff_window + 1:
#             continue

#         # El índice central de la ventana es diff_window
#         # La imagen a procesar es buffer[diff_window]
#         current_img = buffer[diff_window]

#         # Calcular promedio de los diff_window anteriores y posteriores
#         prev_avg = np.mean(list(buffer)[:diff_window], axis=0)
#         next_avg = np.mean(list(buffer)[diff_window+1:], axis=0)
#         diff = np.abs(prev_avg - next_avg)

#         # Umbralizar
#         mean_diff = np.mean(diff)
#         std_diff = np.std(diff)
#         thresh = mean_diff + threshold_sigma * std_diff
#         mask_raw = (diff > thresh).astype(np.uint8) * 255

#         # Limpieza morfológica
#         kernel = np.ones((3, 3), np.uint8)
#         mask_raw = cv2.morphologyEx(mask_raw, cv2.MORPH_OPEN, kernel)
#         mask_raw = cv2.dilate(mask_raw, kernel, iterations=1)

#         # Redimensionar
#         img_resized = cv2.resize(current_img, target_size, interpolation=cv2.INTER_AREA)
#         mask_resized = cv2.resize(mask_raw, target_size, interpolation=cv2.INTER_NEAREST)
#         mask_resized = (mask_resized > 127).astype(np.uint8)

#         # Guardar como .npy
#         entry_id = fits_files[i].stem
#         np.save(output_dir / f"{entry_id}_image.npy", img_resized)
#         np.save(output_dir / f"{entry_id}_mask.npy", mask_resized)

#         # Eliminar la imagen más antigua para liberar memoria (si es necesario)
#         # El deque ya lo hace automáticamente al llegar al maxlen, pero podemos forzar
#         # si queremos liberar antes: buffer.popleft()

#     print(f"✅ Guardados {max(0, len(fits_files) - 2*diff_window)} pares en {output_dir}")


# if __name__ == "__main__":
#     import argparse
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--input", type=Path, required=True)
#     parser.add_argument("--output", type=Path, required=True)
#     parser.add_argument("--size", type=int, default=256)
#     parser.add_argument("--max", type=int, default=200)
#     args = parser.parse_args()
#     generate_pairs(args.input, args.output, (args.size, args.size), args.max)

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
    final_mask_radius: int = 5,  # NUEVO: radio final del objeto en el espacio 256x256, coherente con AnnotationMasking (radius=4-6)
):
    output_dir.mkdir(parents=True, exist_ok=True)
    fits_files = sorted(input_dir.glob("*.fits"))[:max_frames]
    print(f"Procesando {len(fits_files)} archivos...")

    # Usaremos una cola (deque) para mantener solo las últimas imágenes necesarias
    # En lugar de cargar todas, cargamos una ventana de tamaño (2*diff_window + 1)
    # y vamos desechando las antiguas.
    buffer = deque(maxlen=2 * diff_window + 1)  # guarda imágenes en float32

    # NUEVO: contadores de diagnóstico para detectar máscaras vacías/degeneradas
    total_saved = 0
    total_mask_pixels = 0
    empty_masks = 0

    for i, f in enumerate(tqdm(fits_files)):
        with fits.open(f) as hdul:
            data = hdul[0].data.astype(np.float32)
            img = (data - 32768) / 32768.0
            img = np.clip(img, 0, 1)
            buffer.append(img)

        # Solo empezamos a procesar cuando el buffer está lleno
        if len(buffer) < 2 * diff_window + 1:
            continue

        # El índice central de la ventana es diff_window
        # La imagen a procesar es buffer[diff_window]
        current_img = buffer[diff_window]

        # Calcular promedio de los diff_window anteriores y posteriores
        prev_avg = np.mean(list(buffer)[:diff_window], axis=0)
        next_avg = np.mean(list(buffer)[diff_window+1:], axis=0)
        diff = np.abs(prev_avg - next_avg)

        # Umbralizar
        mean_diff = np.mean(diff)
        std_diff = np.std(diff)
        thresh = mean_diff + threshold_sigma * std_diff
        mask_raw = (diff > thresh).astype(np.uint8) * 255

        # Limpieza morfológica
        kernel = np.ones((3, 3), np.uint8)
        mask_raw = cv2.morphologyEx(mask_raw, cv2.MORPH_OPEN, kernel)
        mask_raw = cv2.dilate(mask_raw, kernel, iterations=1)

        # NUEVO: el downscale de ~9600x6422 -> 256x256 es un factor de ~37x por eje.
        # Un streak fino (2-3 px) sobrevive muy raramente a un resize con INTER_NEAREST,
        # porque este simplemente muestrea un punto cada ~37 px y puede caer fuera del streak.
        # Antes de reducir, engordamos el streak en resolución original lo suficiente
        # para que ocupe al menos un "bloque" completo de los que se colapsan en 1 px final.
        scale_x = current_img.shape[1] / target_size[0]
        scale_y = current_img.shape[0] / target_size[1]
        downscale_factor = int(round(max(scale_x, scale_y)))
        prep_kernel_size = max(3, downscale_factor)  # al menos tan grande como el factor de reducción
        prep_kernel = np.ones((prep_kernel_size, prep_kernel_size), np.uint8)
        mask_raw = cv2.dilate(mask_raw, prep_kernel, iterations=1)

        # Redimensionar
        img_resized = cv2.resize(current_img, target_size, interpolation=cv2.INTER_AREA)

        # NUEVO: para la máscara, usamos INTER_AREA (promedio por bloque) en vez de INTER_NEAREST
        # (muestreo puntual). Así, si CUALQUIER píxel positivo cae dentro del bloque que se
        # colapsa a 1 píxel del destino, ese píxel destino recibe un valor > 0 y no se pierde.
        mask_float = mask_raw.astype(np.float32) / 255.0
        mask_resized_float = cv2.resize(mask_float, target_size, interpolation=cv2.INTER_AREA)
        mask_resized = (mask_resized_float > 0.0).astype(np.uint8)

        # NUEVO: un solo píxel positivo en un mapa 256x256 es una señal casi imposible de
        # aprender para el U-Net (Dice/Focal loss se saturan con áreas tan pequeñas).
        # Dilatamos a un radio coherente con el resto del pipeline (AnnotationMasking usa
        # radius=4-6, CircularDynamicMasking usa radius=3-12), para dar un target con área real.
        final_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                                  (2 * final_mask_radius + 1, 2 * final_mask_radius + 1))
        mask_resized = cv2.dilate(mask_resized, final_kernel, iterations=1)

        # NUEVO: diagnóstico de densidad de máscara por frame
        mask_pixels = int(mask_resized.sum())
        total_mask_pixels += mask_pixels
        if mask_pixels == 0:
            empty_masks += 1

        # Guardar como .npy
        entry_id = fits_files[i].stem
        np.save(output_dir / f"{entry_id}_image.npy", img_resized)
        np.save(output_dir / f"{entry_id}_mask.npy", mask_resized)
        total_saved += 1

        # Eliminar la imagen más antigua para liberar memoria (si es necesario)
        # El deque ya lo hace automáticamente al llegar al maxlen, pero podemos forzar
        # si queremos liberar antes: buffer.popleft()

    print(f"✅ Guardados {max(0, len(fits_files) - 2*diff_window)} pares en {output_dir}")
    # NUEVO: resumen de diagnóstico para detectar si el problema de máscaras vacías persiste
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