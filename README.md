# Fast Radon Transform Application

This Python application implements a Fast Radon Transform using FFT (Fast Fourier Transform) and the Fourier Slice Theorem. The Radon transform is commonly used in medical imaging (CT scans), geophysics, and various signal processing applications.

## Features

- Fast implementation of the Radon Transform using FFT techniques
- Interactive GUI built with Tkinter
- Real-time visualization with Matplotlib
- Support for loading custom images or generating test patterns
- Adjustable number of projection angles

## Technical Details

The implementation uses the following approach:
1. Applies padding and circular masking to the input image
2. Performs 2D FFT on the prepared image
3. Extracts slices from the Fourier domain using bilinear interpolation
4. Applies inverse FFT to recover projections for each angle
5. Assembles the projections into a sinogram

The application leverages the Fourier Slice Theorem which states that the 1D Fourier transform of a projection at angle θ is equivalent to a slice at the same angle through the origin of the 2D Fourier transform of the original image.

## Dependencies

- NumPy
- Matplotlib
- SciPy
- Tkinter
- PIL (Pillow)

## Usage

Run the application:

```bash
python radon_transform.py
```

The interface provides options to:
- Load custom images
- Generate a test circle image
- Adjust the number of projection angles
- Execute the Radon transform
- View both the original image and the resulting sinogram

## Applications

The Radon transform is fundamental to:
- Computed Tomography (CT) reconstruction
- Pattern recognition
- Feature extraction
- Image registration
- Motion detection

## Note

For Korean users, the application includes font settings to properly display Korean text in the UI.