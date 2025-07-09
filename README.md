# Vehicle Flow Counting and Statistics Based on OpenCV

This project is designed to count and track vehicles in real-time using OpenCV. It uses background subtraction and morphological operations to detect vehicles and counts them as they pass through a detection line. The project is focused on vehicle flow statistics, ideal for traffic monitoring applications.

## Features

- **Vehicle Detection**: Detects vehicles in video frames using background subtraction (MOG2).
- **Detection Line**: A customizable detection line is drawn to count vehicles passing through it.
- **Real-time Counting**: The number of vehicles that cross the detection line is counted and displayed.
- **Image Processing**: Uses various image processing techniques like Gaussian blur, erosion, dilation, and morphological closing to improve vehicle detection.

## Requirements

- Python 3.x
- OpenCV
- Numpy

## Installation

To install the necessary libraries, you can use `pip`:

```bash
pip install opencv-python opencv-contrib-python numpy
