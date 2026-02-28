#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Descarga de imágenes y archivos .axy desde Astrometry.net.
Primero hace login con API key para obtener cookie de sesión,
luego usa esa sesión para descargar los archivos.
"""

import requests
import os
import time
import shutil
from tqdm import tqdm

# Configuración
API_BASE_URL = "https://nova.astrometry.net/api/"
REFERER_URL = "https://nova.astrometry.net/api/login"
api_key = os.environ.get("ASTROMETRY_API_KEY")
if not api_key:
    raise ValueError("No se encontró la API key en el entorno")

DATASET_PATH = "/app/data/dataset"
JOB_START_ID = 1544128
JOB_AMOUNT = 5000
JOBS_RANGE = range(JOB_START_ID, JOB_START_ID + JOB_AMOUNT)
TIMEOUT_SECONDS = 10
CHUNK_SIZE = 8192

# Crear una sesión que mantendrá las cookies
session = requests.Session()
session.headers.update({"Referer": REFERER_URL})

def login():
    """Realiza login con la API key y obtiene cookie de sesión."""
    login_url = API_BASE_URL + "login"
    # El servidor espera un campo 'request-json' con un JSON que contenga la apikey
    data = {"request-json": f'{{"apikey": "{api_key}"}}'}
    try:
        r = session.post(login_url, data=data, timeout=TIMEOUT_SECONDS)
        r.raise_for_status()
        result = r.json()
        if result.get("status") != "success":
            print("Error en login:", result.get("errormessage", "desconocido"))
            return False
        print("Login exitoso. Sesión iniciada.")
        return True
    except Exception as e:
        print(f"Error en login: {e}")
        return False

def download_file(url, file_path):
    """Descarga un archivo usando la sesión (con cookies)."""
    try:
        with session.get(url, stream=True, timeout=TIMEOUT_SECONDS) as r:
            if r.status_code != 200:
                print(f"Error {r.status_code} al descargar {url}")
                return False

            content_type = r.headers.get("Content-Type", "")
            if "text/html" in content_type:
                print(f"El servidor devolvió HTML (posiblemente requiere autenticación): {url}")
                return False

            total_size = int(r.headers.get("Content-Length", 0))
            with open(file_path, "wb") as f, tqdm(
                total=total_size, unit='B', unit_scale=True,
                desc=os.path.basename(file_path), leave=False
            ) as pbar:
                for chunk in r.iter_content(CHUNK_SIZE):
                    f.write(chunk)
                    pbar.update(len(chunk))
            return True
    except Exception as e:
        print(f"Error en descarga: {e}")
        if os.path.exists(file_path):
            os.remove(file_path)
        return False

def main():
    if not login():
        print("No se pudo iniciar sesión. Abortando.")
        return

    os.makedirs(DATASET_PATH, exist_ok=True)

    for job_id in JOBS_RANGE:
        print(f"\nProcesando trabajo {job_id}...")
        folder = os.path.join(DATASET_PATH, str(job_id))

        # Verificar estado del trabajo (usando la API con la sesión)
        try:
            resp = session.get(
                f"{API_BASE_URL}jobs/{job_id}",
                params={"api_key": api_key},  # opcional, por si acaso
                timeout=TIMEOUT_SECONDS
            )
            resp.raise_for_status()
            job_data = resp.json()
            if job_data.get("status") != "success":
                print(f"⚠️ Job {job_id} no exitoso (estado: {job_data.get('status')})")
                continue
        except Exception as e:
            print(f"Error al consultar estado del job {job_id}: {e}")
            continue

        os.makedirs(folder, exist_ok=True)

        # Descargar archivo .axy
        axy_url = f"https://nova.astrometry.net/axy_file/{job_id}/"
        axy_path = os.path.join(folder, f"{job_id}-axy.fits")
        axy_ok = download_file(axy_url, axy_path)

        # Descargar archivo FITS reescalado (new-fits)
        fits_url = f"https://nova.astrometry.net/new_fits_file/{job_id}/"
        fits_path = os.path.join(folder, f"{job_id}-image.fits")
        fits_ok = download_file(fits_url, fits_path)

        if axy_ok and fits_ok:
            print(f"✅ Job {job_id} completado")
        else:
            print(f"❌ Falló descarga del job {job_id}")
            shutil.rmtree(folder, ignore_errors=True)

        time.sleep(0.2)  # Pausa para no saturar

if __name__ == "__main__":
    main()