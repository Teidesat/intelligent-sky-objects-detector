#!/bin/bash
set -euo pipefail
IFS=$'\n\t'


# This script creates a dataset of sky images with the stars and other objects located in them.
# The dataset is created by downloading images from Astrometry.net website by using web scraping.

# Each dataset entry is composed of:
# _ [entry_id]: Dataset en folder
# |_ [entry_id].fits: The original image
# |_ [entry_id]-axy.fits: The objects located in the image

BASE_WEB_URL="https://nova.astrometry.net"

DATASET_PATH="./dataset"
if [ ! -d "$DATASET_PATH" ]; then
    mkdir -p "$DATASET_PATH"
fi

TEMP_DIR_PATH="./temp"
if [ ! -d "$TEMP_DIR_PATH" ]; then
    mkdir -p "$TEMP_DIR_PATH"
fi


# Function to download the dataset entry information for a given entry ID
download_entry_info() {
    local ENTRY_ID=$1

    # Check if the dataset entry already exists
    if [ -d "$DATASET_PATH/$ENTRY_ID" ] && [ -f "$DATASET_PATH/$ENTRY_ID/$ENTRY_ID.fits" ] && [ -f "$DATASET_PATH/$ENTRY_ID/$ENTRY_ID-axy.fits" ]; then
        echo "Dataset entry $ENTRY_ID already exists, skipping"
        return 0
    fi

    # Download the HTML pages
    local DATASET_ENTRY_URL="$BASE_WEB_URL/user_images/$ENTRY_ID#original"
    local DATASET_ENTRY_HTML_PATH="$TEMP_DIR_PATH/$ENTRY_ID.html"
    if [ -f "$DATASET_ENTRY_HTML_PATH" ]; then
        echo "HTML page $ENTRY_ID already exists, skipping"
    else
        echo "Downloading HTML page $DATASET_ENTRY_URL"
        wget -O "$DATASET_ENTRY_HTML_PATH" "$DATASET_ENTRY_URL" 2>/dev/null || true
    fi

    # Check if the page is valid
    if [ "$(wc -c <"$DATASET_ENTRY_HTML_PATH")" -eq "0" ]; then
        echo "HTML page $ENTRY_ID is empty, aborting"
        # Remove the HTML temp file
        rm "$DATASET_ENTRY_HTML_PATH"
        return 0
    fi

    # Check if the detection job was successful
    local ERROR_MESSAGE
    ERROR_MESSAGE=$(xidel "$DATASET_ENTRY_HTML_PATH" -e '/html/body/div[1]/div[2]/div[1]/div' 2>&1)
    if [ "$ERROR_MESSAGE" != "" ]; then
        echo "Error message: $ERROR_MESSAGE"
        # Remove the HTML temp file
        rm "$DATASET_ENTRY_HTML_PATH"
        return 0
    fi

    # Check if the detection job was successful
    local JOB_STATUS
    JOB_STATUS=$(xidel "$DATASET_ENTRY_HTML_PATH" -e '/html/body/div[1]/div[2]/div[2]/div[1]/div[2]/div/div' 2>&1)
    echo "Job status $ENTRY_ID: $JOB_STATUS"
    if [ "$JOB_STATUS" != "Success" ]; then
        echo "  * aborting entry $ENTRY_ID"
        # Remove the HTML temp file
        rm "$DATASET_ENTRY_HTML_PATH"
        return 0
    fi

    # Create a folder for the dataset entry if it does not exist
    local DATASET_ENTRY_PATH="$DATASET_PATH/$ENTRY_ID"
    if [ ! -d "$DATASET_ENTRY_PATH" ]; then
        mkdir -p "$DATASET_ENTRY_PATH"
    fi

    # Extract the image URL
    local IMAGE_URL
    IMAGE_URL="$BASE_WEB_URL$(xidel "$DATASET_ENTRY_HTML_PATH" -e '/html/body/div[1]/div[2]/div[2]/div[1]/div[3]/table/tbody/tr[9]/td[2]/a/@href' 2>&1)"
    echo "Original image URL: $IMAGE_URL"

    # Download the image
    if [ -f "$DATASET_ENTRY_PATH/$ENTRY_ID.fits" ]; then
        echo "  * image $ENTRY_ID.fits already exists, skipping"
    else
        echo "Downloading entry image $ENTRY_ID"
        wget -O "$DATASET_ENTRY_PATH/$ENTRY_ID.fits" "$IMAGE_URL"
    fi

    # Extract the axy file URL
    local AXY_FILE_URL
    AXY_FILE_URL="$BASE_WEB_URL$(xidel "$DATASET_ENTRY_HTML_PATH" -e '/html/body/div[1]/div[2]/div[2]/div[1]/div[3]/table/tbody/tr[11]/td[2]/a/@href' 2>&1)"
    echo "axy file URL: $AXY_FILE_URL"

    # Download the axy file
    if [ -f "$DATASET_ENTRY_PATH/$ENTRY_ID-axy.fits" ]; then
        echo "  * axy file $ENTRY_ID-axy.fits already exists, skipping"
    else
        echo "Downloading entry axy file $ENTRY_ID"
        wget -O "$DATASET_ENTRY_PATH/$ENTRY_ID-axy.fits" "$AXY_FILE_URL"
    fi

    # Remove the HTML temp file
    rm "$DATASET_ENTRY_HTML_PATH"

    echo "Entry $ENTRY_ID processed"
    return 0
}

# Main function
main() {
    # Get the number of jobs that can be executed in parallel
    NUM_OF_JOBS=$(nproc --ignore=1)
    echo "Executing up to $NUM_OF_JOBS jobs in parallel"

    # Start the loop
    for ENTRY_ID in {1..10};  # Download the entries from 1 to 10
    do
        (
            echo "Processing entry $ENTRY_ID"
            download_entry_info "$ENTRY_ID"
        ) &

        # allow to execute up to $NUM_OF_JOBS jobs in parallel
        if [[ $(jobs -r -p | wc -l) -ge $NUM_OF_JOBS ]]; then
            # now there are $NUM_OF_JOBS jobs already running, so wait here for any
            # job to be finished so there is a place to start next one.
            wait -n
        fi
    done
    wait

    # Remove the temp directory
    rm -r "$TEMP_DIR_PATH"
}

# Execute the main function
main
