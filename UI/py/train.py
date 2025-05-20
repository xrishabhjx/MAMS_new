#!/usr/bin/env python
# coding: utf-8

import math
import cv2
from sklearn import neighbors
import numpy as np
import pandas as pd
import os
import os.path
import pickle
import sys
from PIL import Image
import face_recognition

print("Starting training process...")
curd = os.getcwd()
print("Current directory:", curd)

# Ensure model directory exists
try:  
    os.makedirs(os.path.join(curd, "assets", "models"), exist_ok=True)
    print("Models directory created/verified")
except Exception as e:
    print(f"Error creating models directory: {e}")

# Ensure data directory exists
try:  
    os.makedirs(os.path.join(curd, "assets", "data"), exist_ok=True)
    print("Data directory created/verified")
except Exception as e:
    print(f"Error creating data directory: {e}")

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

# Alternative to face_recognition's image_files_in_folder
def get_image_files(directory):
    """
    Given a directory, return all image files within it
    """
    return [os.path.join(directory, f) for f in os.listdir(directory) 
            if os.path.isfile(os.path.join(directory, f)) and 
            any(f.lower().endswith(ext) for ext in ALLOWED_EXTENSIONS)]

# OpenCV Face detector instead of face_recognition
def detect_face_opencv(image_path):
    """
    Detect faces in an image using OpenCV
    Returns a tuple of face locations (top, right, bottom, left)
    """
    # Load the image
    img = cv2.imread(image_path)
    if img is None:
        print(f"Could not read image: {image_path}")
        return []
    
    # Convert to grayscale for face detection
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Load pre-trained face detector
    face_cascade_path = os.path.join(curd, "py", "haarcascade_frontalface_default.xml")
    if not os.path.exists(face_cascade_path):
        # If not found in py directory, try looking in current directory
        face_cascade_path = "haarcascade_frontalface_default.xml"
        if not os.path.exists(face_cascade_path):
            # Try to copy from opencv install
            cv2_path = os.path.dirname(cv2.__file__)
            data_path = os.path.join(cv2_path, 'data')
            if os.path.exists(data_path):
                cascade_files = [f for f in os.listdir(data_path) if f.startswith('haarcascade_frontalface')]
                if cascade_files:
                    face_cascade_path = os.path.join(data_path, cascade_files[0])
                    print(f"Found cascade file: {face_cascade_path}")
    
    if not os.path.exists(face_cascade_path):
        print("Could not find face cascade file. Using a simpler face detection approach.")
        # Simple face detection - just use the center of the image
        height, width = gray.shape
        face_locations = [(int(height*0.25), int(width*0.75), int(height*0.75), int(width*0.25))]
        return face_locations
    
    face_cascade = cv2.CascadeClassifier(face_cascade_path)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    
    # Convert to face_recognition format (top, right, bottom, left)
    face_locations = []
    for (x, y, w, h) in faces:
        face_locations.append((y, x+w, y+h, x))
    
    return face_locations

# Extract simple features instead of face_recognition's encodings
def extract_face_features(image_path, face_location):
    """
    Extract simple features from a face region
    Returns a feature vector
    """
    # Load image
    img = cv2.imread(image_path)
    if img is None:
        return None
    
    # Extract face region
    top, right, bottom, left = face_location
    face_img = img[top:bottom, left:right]
    
    # Resize to a standard size
    face_img = cv2.resize(face_img, (50, 50))
    
    # Convert to grayscale
    face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
    
    # Histogram of pixel values as a simple feature
    hist = cv2.calcHist([face_img], [0], None, [64], [0, 256])
    hist = cv2.normalize(hist, hist).flatten()
    
    # Add some basic statistics as features
    mean = np.mean(face_img)
    std = np.std(face_img)
    
    # Combine all features
    features = np.concatenate((hist, [mean, std]))
    
    return features

def train(train_dir, model_save_path=None, n_neighbors=None, knn_algo='ball_tree', verbose=True):
    """
    Trains a k-nearest neighbors classifier for face recognition.
    """
    X = []
    y = []

    # Check if training directory exists
    if not os.path.exists(train_dir):
        print(f"Error: Training directory {train_dir} does not exist.")
        sys.exit(1)
    
    # Check if there are any subdirectories (person folders)
    subdirs = [d for d in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, d))]
    if not subdirs:
        print(f"Error: No person directories found in {train_dir}. Please make sure training data is available.")
        sys.exit(1)

    print(f"Found {len(subdirs)} person directories: {', '.join(subdirs)}")
    
    # Loop through each person in the training set
    for class_dir in subdirs:
        class_dir_path = os.path.join(train_dir, class_dir)
        if not os.path.isdir(class_dir_path):
            continue

        # Count usable images
        usable_image_count = 0
        
        # Loop through each training image for the current person
        for img_path in get_image_files(class_dir_path):
            try:
                # Use face_recognition for face detection and encoding
                image = face_recognition.load_image_file(img_path)
                face_locations = face_recognition.face_locations(image)

                if len(face_locations) != 1:
                    if verbose:
                        print(f"Image {img_path} not suitable for training: {'No face detected' if len(face_locations) < 1 else 'Multiple faces detected'}")
                    continue
                else:
                    # Extract features from the face
                    face_encodings = face_recognition.face_encodings(image, face_locations)
                    if len(face_encodings) == 1:
                        X.append(face_encodings[0])
                        y.append(class_dir)
                        usable_image_count += 1
                    else:
                        if verbose:
                            print(f"Image {img_path} not suitable for training: Face encoding failed")
            except Exception as e:
                print(f"Error processing image {img_path}: {e}")
        
        print(f"Person '{class_dir}': {usable_image_count} usable images")
        if usable_image_count == 0:
            print(f"Warning: No usable images found for {class_dir}")

    if len(X) == 0:
        print("Error: No valid training images found. Cannot train model.")
        sys.exit(1)

    # Determine how many neighbors to use for weighting in the KNN classifier
    if n_neighbors is None:
        n_neighbors = int(round(math.sqrt(len(X))))
        if verbose:
            print(f"Chose n_neighbors automatically: {n_neighbors}")

    # Create and train the KNN classifier
    knn_clf = neighbors.KNeighborsClassifier(n_neighbors=n_neighbors, algorithm=knn_algo, weights='distance')
    knn_clf.fit(X, y)

    # Save the trained KNN classifier
    if model_save_path is not None:
        try:
            model_dir = os.path.dirname(model_save_path)
            if not os.path.exists(model_dir):
                os.makedirs(model_dir)
                
            with open(model_save_path, 'wb') as f:
                pickle.dump(knn_clf, f)
            print(f"Model saved to {model_save_path}")
        except Exception as e:
            print(f"Error saving model: {e}")

    return knn_clf

# Main training function
if __name__ == "__main__":
    # Define paths
    curd = os.path.dirname(os.path.abspath(__file__))
    train_dir = os.path.join(curd, "..", "assets", "data")  # Use assets/data relative to project root
    model_path = os.path.join(curd, "..", "assets", "models", "trained_knn_model.clf")
    
    print(f"Training KNN classifier using data from: {train_dir}")
    print(f"Model will be saved to: {model_path}")
    
    try:
        # Train the classifier
        classifier = train(train_dir, model_save_path=model_path, n_neighbors=2)
        print(f"Training complete! Model saved to {model_path}")
    except Exception as e:
        print(f"Error during training: {e}")
        sys.exit(1)

