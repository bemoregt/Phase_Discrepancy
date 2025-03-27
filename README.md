# Phase_Discrepancy

Image subtraction using Fourier Phase Discrepancy technique. This application loads an image, splits it into two halves, and calculates the phase discrepancy between them using Fourier transform techniques.

## Overview

This Python application demonstrates phase-based image comparison using Fourier transforms. The program:
1. Loads an image file and splits it into left and right halves
2. Converts images to grayscale and applies Fourier transforms
3. Extracts amplitude and phase information from each transform
4. Creates complex values based on amplitude differences and phases
5. Applies inverse Fourier transforms and multiplies the results
6. Displays the resulting phase discrepancy visualization

## Requirements
- Python 3.x
- OpenCV (cv2)
- NumPy
- Matplotlib
- Tkinter
- PIL (Pillow)

## Usage
1. Run the application using `python main.py`
2. Click "이미지 로드" to select an image
3. Click "Phase Discrepancy" to process the image and see the results

## Result

![Result](result.png)

## How It Works

The core algorithm works by:
1. Converting input images to grayscale
2. Computing the Fourier transform of each image
3. Extracting phase and amplitude components
4. Creating complex numbers using amplitude differences and original phases
5. Computing inverse Fourier transforms
6. Multiplying the magnitudes of the results to highlight differences

This technique is particularly useful for detecting subtle differences between two very similar images.