import requests
import json
import os
import time
import sys
from tqdm import tqdm  # Import tqdm for the progress bar
import shutil

# Constants
DEFAULT_URL = 'http://nova.astrometry.net'
FITS_FILE_URL = DEFAULT_URL + '/new_fits_file/'
AXY_FILE_URL = DEFAULT_URL + '/axy_file/'
JOB_STATUS_URL = DEFAULT_URL + '/api/jobs/'

DATASET_PATH = './dataset'
JOB_START_ID = 1544128
JOB_AMOUNT = 50
JOBS_RANGE = range(JOB_START_ID, JOB_START_ID + JOB_AMOUNT)
JOB_SUCCESSFUL = "{\"status\": \"success\"}"

# Timeout and speed limits
TIMEOUT_SECONDS = 10  # Max time to wait for a response
MIN_SPEED_BPS = 5000  # Minimum acceptable speed in bytes per second
CHUNK_SIZE = 8192  # 8 KB per chunk

# Ensure dataset folder exists
os.makedirs(DATASET_PATH, exist_ok=True)


def main():
    # Ensure dataset folder exists
    os.makedirs(DATASET_PATH, exist_ok=True)

    # Process each job
    for job_id in JOBS_RANGE:
        print(f"\nProcessing job {job_id}...")

        try:
            status_response = requests.get(JOB_STATUS_URL + str(job_id), timeout=TIMEOUT_SECONDS)
            if status_response.content.decode() == JOB_SUCCESSFUL:
                image_folder = os.path.join(DATASET_PATH, str(job_id))
                os.makedirs(image_folder, exist_ok=True)

                img_path = os.path.join(image_folder, f"{job_id}.fits")
                axy_path = os.path.join(image_folder, f"{job_id}axy.fits")

                axy_success = download_file(AXY_FILE_URL + str(job_id), axy_path)
                if axy_success:
                    img_success = download_file(FITS_FILE_URL + str(job_id), img_path)

                if img_success and axy_success:
                    print(f"\n✅ Successfully downloaded job {job_id}")
                else:
                    print(f"\n❌ Failed to fully download job {job_id}")
                    if os.path.exists(image_folder):
                        shutil.rmtree(image_folder, ignore_errors=True)
            else:
                print(f"\n⚠️ Job {job_id} was not successful")

        except requests.exceptions.RequestException as e:
            print(f"\nFailed to check status for job {job_id}: {e}")

    print("\n✅ Download process completed.")

def download_file(url, file_path):
    """Download a file with a progress bar, timeout, and speed check. Deletes file if incomplete."""
    try:
        with requests.get(url, stream=True, timeout=TIMEOUT_SECONDS) as response:
            if response.status_code != 200:
                print(f"\nFailed to download {url}: HTTP {response.status_code}")
                return False

            # Check file type
            content_type = response.headers.get("Content-Type", "")
            if content_type != "application/fits":
                print(f"\nInvalid file type ({content_type}) for {url}")
                return False

            # Get total file size from headers
            total_size = int(response.headers.get("Content-Length", 0))

            start_time = time.time()
            total_bytes = 0

            with open(file_path, "wb") as file, tqdm(
                total=total_size,
                unit="B",
                unit_scale=True,
                desc=f"Downloading {os.path.basename(file_path)}",
                dynamic_ncols=True,
                leave=False  # Prevents old progress bars from stacking
            ) as progress_bar:
                for chunk in response.iter_content(CHUNK_SIZE):
                    if not chunk:
                        break
                    file.write(chunk)
                    total_bytes += len(chunk)

                    # Update progress bar
                    progress_bar.update(len(chunk))

                    # Calculate and display speed
                    elapsed_time = time.time() - start_time
                    if elapsed_time > 0:
                        speed_kbps = (total_bytes / elapsed_time) / 1024  # Convert to KB/s
                        progress_bar.set_postfix({"Speed": f"{speed_kbps:.2f} KB/s"})

                    # Check speed threshold
                    if elapsed_time > 0 and (total_bytes / elapsed_time) < MIN_SPEED_BPS:
                        print(f"\nDownload too slow ({total_bytes / elapsed_time:.2f} Bps), aborting {file_path}")
                        file.close()
                        os.remove(file_path)  # Delete partial file
                        return False

            print(f"\nDownloaded {file_path} ({total_bytes} bytes)")
            return True

    except requests.exceptions.Timeout:
        print(f"\nTimeout while downloading {url}")
    except requests.exceptions.RequestException as e:
        print(f"\nError downloading {url}: {e}")

    # Cleanup in case of failure
    if os.path.exists(file_path):
        os.remove(file_path)
    return False


if __name__ == "__main__":
    main()
