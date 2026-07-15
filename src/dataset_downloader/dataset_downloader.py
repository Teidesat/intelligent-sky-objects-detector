#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Downloads images (.fits), sources (.axy) and annotations (.json) from Astrometry.net.
Re-runnable: skips already downloaded files and only downloads missing ones.
"""


import json
import requests
import os
import time
import shutil
from pathlib import Path
from tqdm import tqdm

API_BASE_URL  = "https://nova.astrometry.net/api/"
REFERER_URL   = "https://nova.astrometry.net/api/login"
api_key = os.environ.get("ASTROMETRY_API_KEY")
if not api_key:
    raise ValueError("API key not found. Please set the ASTROMETRY_API_KEY environment variable.")
DATASET_PATH    = Path("/app/data/dataset")
JOB_START_ID    = 1544001 # 11000000 ### 27 # 110
JOB_AMOUNT      = 4240 # 1467 ### 1 # 1
JOBS_RANGE      = range(JOB_START_ID, JOB_START_ID + JOB_AMOUNT)
TIMEOUT_SECONDS = 10
CHUNK_SIZE      = 8192
STAR_TYPES      = {"hd", "tycho2", "twomass", "usnob"}

session = requests.Session()
session.headers.update({"Referer": REFERER_URL})


def login() -> bool:
    url  = API_BASE_URL + "login"
    data = {"request-json": f'{{"apikey": "{api_key}"}}'}
    try:
        r = session.post(url, data=data, timeout=TIMEOUT_SECONDS)
        r.raise_for_status()
        result = r.json()
        if result.get("status") != "success":
            print("Error in login:", result.get("errormessage"))
            return False
        print("Login successful.")
        return True
    except Exception as e:
        print(f"Error in login: {e}")
        return False


def check_job_status(job_id: int) -> bool:
    try:
        resp = session.get(
            f"{API_BASE_URL}jobs/{job_id}",
            params={"api_key": api_key},
            timeout=TIMEOUT_SECONDS,
        )
        resp.raise_for_status()
        return resp.json().get("status") == "success"
    except Exception as e:
        print(f"  Error checking job status {job_id}: {e}")
        return False


def download_file(url: str, file_path: Path) -> bool:
    try:
        with session.get(url, stream=True, timeout=TIMEOUT_SECONDS) as r:
            if r.status_code != 200:
                print(f"  HTTP {r.status_code}: {url}")
                return False
            if "text/html" in r.headers.get("Content-Type", ""):
                print(f"  HTML Response (?: {url}")
                return False
            total = int(r.headers.get("Content-Length", 0))
            with open(file_path, "wb") as f, tqdm(
                total=total, unit="B", unit_scale=True,
                desc=file_path.name, leave=False,
            ) as pbar:
                for chunk in r.iter_content(CHUNK_SIZE):
                    f.write(chunk)
                    pbar.update(len(chunk))
        return True
    except Exception as e:
        print(f"  Error in download: {e}")
        if file_path.exists():
            file_path.unlink()
        return False


def download_annotations(job_id: int, ann_path: Path) -> bool:
    """Download only star-type annotations (HD, Tycho-2, 2MASS, USNO-B)."""
    url = f"{API_BASE_URL}jobs/{job_id}/annotations/"
    try:
        r = session.get(url, timeout=TIMEOUT_SECONDS)
        r.raise_for_status()
        all_ann = r.json().get("annotations", [])
        stars   = [a for a in all_ann if a.get("type") in STAR_TYPES]
        with open(ann_path, "w") as f:
            json.dump(stars, f)
        return True
    except Exception as e:
        print(f"  Error downloading annotations for job {job_id}: {e}")
        return False


def main():
    if not login():
        print("Failed to log in. Aborting.")
        return

    DATASET_PATH.mkdir(parents=True, exist_ok=True)

    for job_id in JOBS_RANGE:
        folder   = DATASET_PATH / str(job_id)
        fits_path = folder / f"{job_id}-image.fits"
        axy_path  = folder / f"{job_id}-axy.fits"
        ann_path  = folder / f"{job_id}-annotations.json"

        need_fits = not fits_path.exists()
        need_axy  = not axy_path.exists()
        need_ann  = not ann_path.exists()

        if not need_fits and not need_axy and not need_ann:
            continue

        print(f"\nJob {job_id} — faltan: "
              f"{'image ' if need_fits else ''}"
              f"{'axy ' if need_axy else ''}"
              f"{'annotations' if need_ann else ''}")

        if need_fits or need_axy:
            if not check_job_status(job_id):
                print(f"  WARNING: Job failed, skipping.")
                continue

        folder.mkdir(exist_ok=True)

        if need_fits:
            url = f"https://nova.astrometry.net/new_fits_file/{job_id}/"
            if not download_file(url, fits_path):
                print(f"  X Failed to download image.fits")
                if not axy_path.exists():
                    shutil.rmtree(folder, ignore_errors=True)
                continue

        if need_axy:
            url = f"https://nova.astrometry.net/axy_file/{job_id}/"
            if not download_file(url, axy_path):
                print(f"  X Failed to download axy.fits")
                continue

        if need_ann:
            ok = download_annotations(job_id, ann_path)
            n_stars = len(json.load(open(ann_path))) if ok and ann_path.exists() else 0
            print(f"  {'OK' if ok else 'WARNING:'} annotations: {n_stars} catalog stars")
        else:
            pass

        time.sleep(0.15)


if __name__ == "__main__":
    main()