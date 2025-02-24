#!/bin/bash
set -o nounset -o errexit -o pipefail
IFS=$'\n\t'


#
# Use dataset-downloader.py as it works better
#




# This script creates a dataset of sky images with the stars and other objects located in them.
# The dataset is created by downloading images from Astrometry.net website by using web scraping.

# Each dataset entry is composed of:
# _ [entry_id]: Dataset en folder
# |_ [entry_id].fits: The original image
# |_ [entry_id]-axy.fits: The objects located in the image

BASE_WEB_URL="https://nova.astrometry.net"
DATASET_PATH="./dataset"
TEMP_DIR_PATH="./temp"


# Function to download the dataset entry information for a given entry ID
download_entry_info() {
    local ENTRY_ID=$1

    # Create the dataset and temp directories if they do not exist
    if [ ! -d "$DATASET_PATH" ]; then
        mkdir --parents "$DATASET_PATH"
    fi
    if [ ! -d "$TEMP_DIR_PATH" ]; then
        mkdir --parents "$TEMP_DIR_PATH"
    fi

    # Check if the dataset entry already exists
    local DATASET_ENTRY_PATH="$DATASET_PATH/$ENTRY_ID"
    if [ -d "$DATASET_ENTRY_PATH" ] && [ -f "$DATASET_ENTRY_PATH/$ENTRY_ID.fits" ] && [ -f "$DATASET_ENTRY_PATH/$ENTRY_ID-axy.fits" ]; then
        echo "Dataset entry $ENTRY_ID already exists, aborting"
        return 0
    fi

    # Download the HTML pages
    local DATASET_ENTRY_URL="$BASE_WEB_URL/user_images/$ENTRY_ID#original"
    local DATASET_ENTRY_HTML_PATH="$TEMP_DIR_PATH/$ENTRY_ID.html"

    if [ -f "$DATASET_ENTRY_HTML_PATH" ]; then
        echo "HTML page $DATASET_ENTRY_HTML_PATH already exists, skipping download"

    else
        echo "Downloading HTML page $DATASET_ENTRY_URL"
        curl --silent --fail --output "$DATASET_ENTRY_HTML_PATH" "$DATASET_ENTRY_URL" 2>/dev/null

        # Check if the HTML page was downloaded
        if [ ! -f "$DATASET_ENTRY_HTML_PATH" ]; then
            echo "HTML page $DATASET_ENTRY_HTML_PATH could not be downloaded, aborting entry $ENTRY_ID"
            return 0
        fi
    fi

    # Check if the page is valid
    if [ "$(wc --chars <"$DATASET_ENTRY_HTML_PATH")" -eq "0" ]; then
        echo "HTML page $DATASET_ENTRY_HTML_PATH is empty, aborting entry $ENTRY_ID"

        # Remove the HTML temp file
        rm "$DATASET_ENTRY_HTML_PATH"
        return 0
    fi

    # Check if the detection job was successful
    local ERROR_MESSAGE
    ERROR_MESSAGE=$(xidel --silent "$DATASET_ENTRY_HTML_PATH" --extract="/html/body/div[1]/div[2]/div[1]/div" 2>&1)

    if [ "$ERROR_MESSAGE" != "" ]; then
        echo "Error message: $ERROR_MESSAGE, aborting entry $ENTRY_ID"

        # Remove the HTML temp file
        rm "$DATASET_ENTRY_HTML_PATH"
        return 0
    fi

    # Check if the detection job was successful
    local JOB_STATUS
    JOB_STATUS=$(xidel --silent "$DATASET_ENTRY_HTML_PATH" --extract="/html/body/div[1]/div[2]/div[2]/div[1]/div[2]/div/div" 2>&1)
    echo "Job status $ENTRY_ID: $JOB_STATUS"

    if [ "$JOB_STATUS" != "Success" ]; then
        echo "  * aborting entry $ENTRY_ID"

        # Remove the HTML temp file
        rm "$DATASET_ENTRY_HTML_PATH"
        return 0
    fi

    # Create a folder for the dataset entry if it does not exist
    if [ ! -d "$DATASET_ENTRY_PATH" ]; then
        mkdir --parents "$DATASET_ENTRY_PATH"
    fi

    # Extract the image URL
    local IMAGE_URL
    IMAGE_URL="$BASE_WEB_URL$(xidel --silent "$DATASET_ENTRY_HTML_PATH" --extract="/html/body/div[1]/div[2]/div[2]/div[1]/div[3]/table/tbody/tr[9]/td[2]/a/@href" 2>&1)"
    echo "Original image URL of entry $ENTRY_ID: $IMAGE_URL"

    # Download the image
    if [ -f "$DATASET_ENTRY_PATH/$ENTRY_ID.fits" ]; then
        echo "  * image $ENTRY_ID.fits already exists, skipping download"

    else
        echo "Downloading entry image $ENTRY_ID"
        curl --silent --fail --output "$DATASET_ENTRY_PATH/$ENTRY_ID.fits" "$IMAGE_URL"

        # Check if the image was downloaded
        if [ ! -f "$DATASET_ENTRY_PATH/$ENTRY_ID.fits" ]; then
            echo "Image $ENTRY_ID.fits could not be downloaded, dataset entry $ENTRY_ID will be incomplete"
        elif grep --quiet --extended-regexp "Error|Failed" "$DATASET_ENTRY_PATH/$ENTRY_ID.fits"; then
            echo "Image $ENTRY_ID.fits is not a valid FITS image file, dataset entry $ENTRY_ID will be incomplete"
            rm "$DATASET_ENTRY_PATH/$ENTRY_ID.fits"
        fi

        # If image download was not successful
        if [ ! -f "$DATASET_ENTRY_PATH/$ENTRY_ID.fits" ]; then
            PREVIEW_IMAGE_URL="$BASE_WEB_URL$(xidel --silent "$DATASET_ENTRY_HTML_PATH" --extract="//*[@id="image_container"]" 2>&1)"
            curl --silent --fail --output "$DATASET_ENTRY_PATH/$ENTRY_ID.fits" "$PREVIEW_IMAGE_URL"

            # Check if the image was downloaded
            if [ ! -f "$DATASET_ENTRY_PATH/$ENTRY_ID.fits" ]; then
                echo "Image $ENTRY_ID.fits could not be downloaded, dataset entry $ENTRY_ID will be incomplete"
            elif grep --quiet --extended-regexp "Error|Failed" "$DATASET_ENTRY_PATH/$ENTRY_ID.fits"; then
                echo "Image $ENTRY_ID.fits is not a valid FITS image file, dataset entry $ENTRY_ID will be incomplete"
                rm "$DATASET_ENTRY_PATH/$ENTRY_ID.fits"
            fi
        fi
    fi

    # Extract the axy file URL
    local AXY_FILE_URL
    AXY_FILE_URL="$BASE_WEB_URL$(xidel --silent "$DATASET_ENTRY_HTML_PATH" --extract="/html/body/div[1]/div[2]/div[2]/div[1]/div[3]/table/tbody/tr[11]/td[2]/a/@href" 2>&1)"
    echo "axy file URL of entry $ENTRY_ID: $AXY_FILE_URL"

    # Download the axy file
    if [ -f "$DATASET_ENTRY_PATH/$ENTRY_ID-axy.fits" ]; then
        echo "  * axy file $ENTRY_ID-axy.fits already exists, skipping download"

    else
        echo "Downloading entry axy file $ENTRY_ID"
        curl --silent --fail --output "$DATASET_ENTRY_PATH/$ENTRY_ID-axy.fits" "$AXY_FILE_URL"

        # Check if the axy file was downloaded
        if [ ! -f "$DATASET_ENTRY_PATH/$ENTRY_ID-axy.fits" ]; then
            echo "axy file $ENTRY_ID-axy.fits could not be downloaded, dataset entry $ENTRY_ID will be incomplete"
        fi
    fi

    # Remove the HTML temp file
    rm "$DATASET_ENTRY_HTML_PATH"

    # Remove dataset entry if it is empty
    if [ -d "$DATASET_ENTRY_PATH" ] && [ ! "$(ls --almost-all "$DATASET_ENTRY_PATH")" ]; then
        echo "Dataset entry $ENTRY_ID is empty, removing it"
        rm --dir "$DATASET_ENTRY_PATH"
    fi

    echo "Entry $ENTRY_ID processing completed"
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
            download_entry_info "$ENTRY_ID" || true  # Ignore errors to continue with the next dataset entry
        ) &

        # allow to execute up to $NUM_OF_JOBS jobs in parallel
        if [[ $(jobs -r -p | wc --lines) -ge $NUM_OF_JOBS ]]; then
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
