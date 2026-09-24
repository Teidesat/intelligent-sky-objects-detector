#!/bin/bash
set -o nounset -o errexit -o pipefail
IFS=$'\n\t'


# This script creates a dataset of sky images with the stars and other objects located in them.
# The dataset is created by downloading images from Astrometry.net website by using web scraping.

# Each dataset entry is composed of:
# _ [entry_id]: Dataset en folder
# |_ [entry_id].fits: The original image
# |_ [entry_id]-axy.fits: The objects located in the image

BASE_WEB_URL="https://nova.astrometry.net"
JOB_STATUS_URL="$BASE_WEB_URL/api/jobs"
IMAGE_FILE_URL="$BASE_WEB_URL/new_fits_file"
AXY_FILE_URL="$BASE_WEB_URL/axy_file"

DATASET_PATH="./dataset"
TEMP_DIR_PATH="./temp"


# Function to download the dataset entry information for a given entry ID
download_entry_info() {
    local ENTRY_ID=$1

    # Create the dataset and temp directories if they do not exist
    if [ ! -d "$DATASET_PATH" ]; then
        mkdir --parents "$DATASET_PATH"
    fi

    # Check if the dataset entry already exists
    local DATASET_ENTRY_PATH="$DATASET_PATH/$ENTRY_ID"
    local IMAGE_FILE_PATH="$DATASET_ENTRY_PATH/$ENTRY_ID-image.fits"
    local AXY_FILE_PATH="$DATASET_ENTRY_PATH/$ENTRY_ID-axy.fits"
    if [ -d "$DATASET_ENTRY_PATH" ] && [ -f "$IMAGE_FILE_PATH" ] && [ -f "$AXY_FILE_PATH" ]; then
        echo "Dataset entry $ENTRY_ID already exists, aborting"
        return 0
    fi

    # Check if the detection job was successful
    local JOB_STATUS
    JOB_STATUS=$(curl --silent --fail --request GET "$JOB_STATUS_URL/$ENTRY_ID")
    echo "Job status $ENTRY_ID: $JOB_STATUS"

    if [ "$JOB_STATUS" != '{"status": "success"}' ]; then
        echo "  * aborting entry $ENTRY_ID"
        return 0
    fi

    # Create a folder for the dataset entry if it does not exist
    if [ ! -d "$DATASET_ENTRY_PATH" ]; then
        mkdir --parents "$DATASET_ENTRY_PATH"
    fi

    # Download the image
    if [ -f "$IMAGE_FILE_PATH" ]; then
        echo "  * image $ENTRY_ID-image.fits already exists, skipping download"

    else
        echo "Downloading entry image $ENTRY_ID"
        curl --silent --fail --output "$IMAGE_FILE_PATH" --request GET "$IMAGE_FILE_URL/$ENTRY_ID"

        # Check if the image was downloaded
        if [ ! -f "$IMAGE_FILE_PATH" ]; then
            echo "Image $ENTRY_ID-image.fits could not be downloaded, dataset entry $ENTRY_ID will be incomplete"
        elif grep --quiet --extended-regexp "Error|Failed" "$IMAGE_FILE_PATH"; then
            echo "Image $ENTRY_ID-image.fits is not a valid FITS image file, dataset entry $ENTRY_ID will be incomplete"
            rm "$IMAGE_FILE_PATH"
        fi
    fi

    # Download the axy file
    if [ -f "$AXY_FILE_PATH" ]; then
        echo "  * axy file $ENTRY_ID-axy.fits already exists, skipping download"

    else
        echo "Downloading entry axy file $ENTRY_ID"
        curl --silent --fail --output "$AXY_FILE_PATH" --request GET "$AXY_FILE_URL/$ENTRY_ID"

        # Check if the axy file was downloaded
        if [ ! -f "$AXY_FILE_PATH" ]; then
            echo "axy file $ENTRY_ID-axy.fits could not be downloaded, dataset entry $ENTRY_ID will be incomplete"
        elif grep --quiet --extended-regexp "Error|Failed" "$AXY_FILE_PATH"; then
            echo "axy file $ENTRY_ID-axy.fits is not a valid FITS image file, dataset entry $ENTRY_ID will be incomplete"
            rm "$AXY_FILE_PATH"
        fi
    fi

    # Remove dataset entry if it is empty
    if [ -d "$DATASET_ENTRY_PATH" ] && [ ! "$(ls --almost-all "$DATASET_ENTRY_PATH")" ]; then
        echo "Dataset entry $ENTRY_ID is empty, removing it"
        rm --dir "$DATASET_ENTRY_PATH"
    fi

    echo "Entry $ENTRY_ID processing completed"
    return 0
}

get_random_entry_id() {
    local MIN_ENTRY_ID=$1
    local MAX_ENTRY_ID=$2

    python -c "import random as R; print(R.randint($MIN_ENTRY_ID, $MAX_ENTRY_ID))"
}

# Main function
main() {
    # Get the number of jobs that can be executed in parallel
    NUM_OF_JOBS=$(nproc --ignore=1)
    echo "Executing up to $NUM_OF_JOBS jobs in parallel"

    # Start the loop
    for ITERATION_INDEX in {1..10}; do  # Try to download 10 entries
        (
            ENTRY_ID=$(get_random_entry_id 1 10000000)  # Get a random entry ID between 1 and 10000000
            echo "Processing entry $ENTRY_ID"
            download_entry_info "$ENTRY_ID" || true  # Ignore errors to continue with the next dataset entry
        ) &  # Process the entry in the background

        # allow to execute up to $NUM_OF_JOBS jobs in parallel
        if [[ $(jobs -r -p | wc --lines) -ge $NUM_OF_JOBS ]]; then
            # now there are $NUM_OF_JOBS jobs already running, so wait here for any
            # job to be finished so there is a place to start next one.
            wait -n
        fi
    done
    wait
}

# Execute the main function
main
