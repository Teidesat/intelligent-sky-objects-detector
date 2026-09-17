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

from pathlib import Path
import numpy as np
from .entry import DatasetEntry

class FrigatePairLoader:
    """
    Loader for Frigate-generated image-mask pairs. 
    It loads .npy files containing images and their corresponding segmentation masks, 
    and returns a dictionary of DatasetEntry objects.
    """
    def __init__(self, target_shape=(256, 256)):
        self.target_shape = target_shape

    def load(self, dataset_path: Path) -> dict[str, DatasetEntry]:
        dataset = {}
        for img_path in dataset_path.glob("*_image.npy"):
            entry_id = img_path.stem.replace("_image", "")
            mask_path = dataset_path / f"{entry_id}_mask.npy"
            if not mask_path.exists():
                continue
            img = np.load(img_path)
            mask = np.load(mask_path)
            dataset[entry_id] = DatasetEntry(
                entry_id=entry_id,
                nn_input_image=img,
                segmentation_mask=mask,
                filtered_objects=[]
            )
        print(f"Loaded {len(dataset)} Frigate pairs")
        return dataset