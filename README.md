# Intelligent Sky Objects Detector

## Introduction

This project is a part of the Teidesat Cubesat Tracker. The main goal of this project is to train a Convolutional Neural Network (CNN) to detect bright point-like objects in images of the night sky; like stars, satellites, or airplanes.

The CNN will be trained with images of the sky where previous astrometry has been performed to mark the position of the present stars. The resulting model will be able to provide the positions of the sky objects at the given images.


## Dataset

The dataset used for the training phase was obtained from [Astrometry.net](https://nova.astrometry.net/) web page, which has lots of images of the night sky with astrometry already performed on each, providing different information about the image and its content, like the position of the stars, the coordinates and orientation of the image, etc.

The dataset is composed of numerous images of the night sky, with different sizes and resolutions, and with different numbers of stars in each image. Each dataset entry is composed of two FITS files: one with the image of the sky and another with the stars' positions.


## Data Preprocessing

Before training the CNN, the images are preprocessed to convert them to grayscale, equalize their sizes, and normalize their pixel values; and the stars' positions are extracted from the FITS files and used to create a binary mask for each image, in which the pixels where the stars are located are marked with a 1 and the rest with a 0 as background.


## Model

The CNN model used for this project follows a U-Net architecture, which was developed for image segmentation. The model is composed of an encoder and a decoder, with skip connections between them to help the model to learn the features of the images at different scales.


## Input and Output

The input of the resulting model has to be a monochrome image of the night sky with a size equal to the one used for training.

The output will be a binary mask with the same size as the input image, where the pixels with a 1 represent the position of the stars in the image.

To obtain a list of the positions of the stars in the image, the binary mask needs to be processed as follows:
1. Find the contours of the binary mask to group the pixels that belong to the same star.
2. Compute the centroid of each contour to get the position of the star in the image.
