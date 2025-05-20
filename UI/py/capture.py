#!/usr/bin/env python
# coding: utf-8

import cv2
import numpy as np
import time
import os
import sys

# Get the current directory and print for debugging
curd = os.getcwd()
print("Starting webcam capture process")
print("Current directory:", curd)

# Create assets and data directories if they don't exist
try:
    os.makedirs(os.path.join(curd, "assets", "data"), exist_ok=True)
except OSError as e:
    print(f"Error creating directories: {e}")
    
# Read the student registration number from helper.txt
try:
    with open("py/helper.txt", "r") as f:
        name = f.read().strip()
    print(f"Student registration number: {name}")
except Exception as e:
    print(f"Error reading helper.txt: {e}")
    sys.exit(1)

# Create directory for this student's images
path = os.path.join(curd, "assets", "data", name)
try:
    os.makedirs(path, exist_ok=True)
    print(f"Directory created/verified: {path}")
except OSError as e:
    print(f"Error creating student directory: {e}")

# Initialize webcam - try multiple camera indices if needed
camera_index = 0
cap = None

while camera_index < 3:  # Try first 3 camera indices
    try:
        cap = cv2.VideoCapture(camera_index)
        if cap.isOpened():
            print(f"Successfully opened camera with index {camera_index}")
            break
        camera_index += 1
    except Exception as e:
        print(f"Error opening camera {camera_index}: {e}")
        camera_index += 1

if cap is None or not cap.isOpened():
    print("Failed to open any camera. Please make sure your webcam is connected and not in use by another application.")
    sys.exit(1)

# Set camera properties
cap.set(cv2.CAP_PROP_FPS, 30)

# Warm up the camera
print("Warming up camera...")
for i in range(5):
    ret, frame = cap.read()
    time.sleep(0.1)

# Capture 10 images
count = 0
print("Starting to capture 10 images...")

while count < 10:
    ret, frame = cap.read()
    
    if not ret or frame is None:
        print("Failed to grab frame")
        continue
        
    # Save the image without displaying
    image_path = os.path.join(path, f"{name}{count}.jpg")
    try:
        cv2.imwrite(image_path, frame)
        print(f"Saved image {count+1}/10: {image_path}")
        count += 1
    except Exception as e:
        print(f"Error saving image: {e}")
    
    # Wait - slower capture to avoid similar images
    time.sleep(1)

# Cleanup
cap.release()
print(f"Capture complete! {count} images saved to {path}")

