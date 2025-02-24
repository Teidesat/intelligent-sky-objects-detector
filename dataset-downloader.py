import requests
import json
import os
import time
import sys

# Constants
DEFAULT_URL = 'http://nova.astrometry.net'
FITS_FILE_URL = DEFAULT_URL + '/new_fits_file/'
AXY_FILE_URL = DEFAULT_URL + '/axy_file/'
JOB_STATUS_URL = DEFAULT_URL + '/api/jobs/'

DATASET_PATH = './dataset'
JOB_START_ID = 1444114
JOB_AMOUNT = 10
JOBS_RANGE = range(JOB_START_ID, JOB_START_ID + JOB_AMOUNT)
JOB_SUCCESSFUL = "{\"status\": \"success\"}"

# Timeout and speed limits
TIMEOUT_SECONDS = 10  # Max time to wait for a response
MIN_SPEED_BPS = 5000  # Minimum acceptable speed in bytes per second
CHUNK_SIZE = 8192  # 8 KB per chunk

# Ensure dataset folder exists
os.makedirs(DATASET_PATH, exist_ok=True)

def download_file(url, file_path):
    """Download a file with a timeout and speed check, delete if incomplete."""
    try:
        with requests.get(url, stream=True, timeout=TIMEOUT_SECONDS) as response:
            if response.status_code != 200:
                print(f"Failed to download {url}: HTTP {response.status_code}")
                return False

            # Check file type
            content_type = response.headers.get("Content-Type", "")
            if content_type != "application/fits":
                print(f"Invalid file type ({content_type}) for {url}")
                return False

            # Start the download with speed check
            start_time = time.time()
            total_bytes = 0

            with open(file_path, "wb") as file:
                for chunk in response.iter_content(CHUNK_SIZE):
                    if not chunk:  # Connection lost
                        break
                    file.write(chunk)
                    total_bytes += len(chunk)

                    # Check speed
                    elapsed_time = time.time() - start_time
                    if elapsed_time > 0:
                        speed_kbps = (total_bytes / elapsed_time) / 1024  # Convert to KB/s
                        sys.stdout.write(f"\rDownloading {file_path} - Speed: {speed_kbps:.2f} KB/s")
                        sys.stdout.flush()

                    if elapsed_time > 0 and (total_bytes / elapsed_time) < MIN_SPEED_BPS:
                        print(f"Download too slow ({total_bytes / elapsed_time:.2f} Bps), aborting {file_path}")
                        file.close()
                        os.remove(file_path)  # Delete partial file
                        return False

            print(f"Downloaded {file_path} ({total_bytes} bytes\n)")
            return True

    except requests.exceptions.Timeout:
        print(f"Timeout while downloading {url}")
    except requests.exceptions.RequestException as e:
        print(f"Error downloading {url}: {e}")

    # Cleanup in case of failure
    if os.path.exists(file_path):
        os.remove(file_path)
    return False

# Process each job
for job_id in JOBS_RANGE:
    print(f"Processing job {job_id}...")

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
                print(f"Successfully downloaded job {job_id}")
            else:
                print(f"Failed to fully download job {job_id}")
                if os.path.exists(image_folder):
                  os.remove(image_folder)
        else:
            print(f"Job {job_id} was not successful")

    except requests.exceptions.RequestException as e:
        print(f"Failed to check status for job {job_id}: {e}")

print("Download process completed.")