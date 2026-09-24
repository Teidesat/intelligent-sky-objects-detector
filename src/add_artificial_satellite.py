import astropy.io.fits as fits
import cv2 as cv
import numpy as np
import os

# ID de la carpeta actual
JOB_ID = "1544485" 

input_path = f"/app/data/dataset/{JOB_ID}/{JOB_ID}-image.fits"
output_path = f"/app/data/dataset/{JOB_ID}/test_satelite_saturado.fits"

print(f"Intentando abrir: {input_path}")

if not os.path.exists(input_path):
    raise FileNotFoundError(f"El archivo base no existe. Verifica si el ID {JOB_ID} está descargado.")

# 1. Leer imagen original y sus metadatos
with fits.open(input_path) as hdul:
    header = hdul[0].header.copy()
    original_shape = hdul[0].data.shape
    print(f"Forma original del FITS: {original_shape}")
    
    # Convertimos a float32 nativo (corrige Big-Endian)
    data = hdul[0].data.astype(np.float32)

# 2. Adaptar dimensiones para OpenCV si es una imagen (Canales, Alto, Ancho)
is_channels_first = data.ndim == 3 and data.shape[0] < data.shape[1]

if is_channels_first:
    # Pasamos de (Canales, Alto, Ancho) -> (Alto, Ancho, Canales)
    data = np.transpose(data, (1, 2, 0))
    # ¡CRÍTICO! Reorganiza los datos físicamente en la memoria RAM para OpenCV
    data = np.ascontiguousarray(data)
    print(f"Forma reordenada para OpenCV: {data.shape}")
    num_channels = data.shape[2]
else:
    num_channels = 1 if data.ndim == 2 else data.shape[2]

# 3. Inyectar el satélite artificial (Saturación Máxima)
valor_saturacion = 65535.0 

# Si tiene múltiples canales, definimos el color para todos ellos
color_multicanal = tuple([valor_saturacion] * num_channels) if data.ndim == 3 else valor_saturacion

# Dibujar el punto saturado (Ahora OpenCV tendrá la memoria limpia y ordenada)
cv.circle(data, (128, 128), radius=8, color=color_multicanal, thickness=-1)

# Aplicar el difuminado para simular el blooming
data = cv.GaussianBlur(data, (5, 5), 0) 

# 4. Devolver al formato astronómico original antes de guardar
if is_channels_first:
    # Pasamos de (Alto, Ancho, Canales) -> (Canales, Alto, Ancho)
    data = np.transpose(data, (2, 0, 1))
    data = np.ascontiguousarray(data)
    print(f"Forma restaurada para el FITS: {data.shape}")

# Guardar el FITS modificado de forma segura
fits.writeto(output_path, data, header=header, overwrite=True)

print(f"✅ Satélite sintético inyectado con éxito en: {output_path}")