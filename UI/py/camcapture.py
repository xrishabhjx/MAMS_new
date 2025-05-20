#!/usr/bin/env python
# coding: utf-8

import cv2
import numpy as np
import pandas as pd
import os
import pickle
import time
import sys
import face_recognition

if len(sys.argv) > 1:
    day = sys.argv[1]  # e.g., '2' for Day2
else:
    day = '1'  # default to Day1

# Use 'Day' + day as the column name to update
column_name = 'Day' + day

print("Starting face recognition for attendance...")
curd = os.getcwd()

# Function to load the model
def load_model(model_path):
    try:
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
            return model
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

# Function to extract face features (same as in train.py)
def extract_face_features(image, face_location):
    """
    Extract simple features from a face region
    Returns a feature vector
    """
    # Extract face region
    top, right, bottom, left = face_location
    face_img = image[top:bottom, left:right]
    
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

# Function to detect faces using OpenCV
def detect_faces(image):
    # Convert to grayscale for face detection
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Load pre-trained face detector
    face_cascade_path = os.path.join(curd, "py", "haarcascade_frontalface_default.xml")
    if not os.path.exists(face_cascade_path):
        print(f"Error: Could not find cascade file at {face_cascade_path}")
        return []
    
    face_cascade = cv2.CascadeClassifier(face_cascade_path)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    
    # Convert to format (top, right, bottom, left)
    face_locations = []
    for (x, y, w, h) in faces:
        face_locations.append((y, x+w, y+h, x))
    
    return face_locations

# Function to recognize faces
def recognize_faces(frame, model):
    # Convert BGR (OpenCV) to RGB (face_recognition)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    face_locations = face_recognition.face_locations(rgb_frame)
    if not face_locations:
        return []
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
    if not face_encodings:
        return []
    closest_distances = model.kneighbors(face_encodings, n_neighbors=1)
    are_matches = [closest_distances[0][i][0] <= 0.5 for i in range(len(face_locations))]
    predictions = []
    for pred, loc, rec in zip(model.predict(face_encodings), face_locations, are_matches):
        name = pred if rec else "unknown"
        predictions.append((name, loc))
    return predictions

# Start headless webcam capture for attendance
def take_attendance():
    model_path = os.path.join(curd, "assets", "models", "trained_knn_model.clf")
    model = load_model(model_path)
    
    if model is None:
        print("Could not load model. Please train the model first.")
        return
    
    detected_names = []
    
    # Initialize webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Could not open webcam")
        return
    
    print("Starting webcam for attendance. Will capture for 30 seconds.")
    print("Keep your face in front of the camera.")
    
    # Capture for 30 seconds
    end_time = time.time() + 30
    frame_count = 0
    
    while time.time() < end_time:
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture frame")
            break
        
        frame_count += 1
        if frame_count % 5 != 0:  # Process every 5th frame
            continue

        # Recognize faces
        predictions = recognize_faces(frame, model)
        
        # Add names to attendance list
        for name, _ in predictions:
            if name != "unknown":
                detected_names.append(name)
                print(f"Detected: {name}")
        
        # Small pause
        time.sleep(0.2)
    
    # Cleanup
    cap.release()
    
    print(f"Attendance capture complete. Processed {frame_count} frames.")
    
    # Create attendance record
    if detected_names:
        # Count occurrences of each name
        name_counts = pd.Series(detected_names).value_counts()
        
        # Create attendance DataFrame
        attendance = pd.DataFrame(name_counts)
        attendance.columns = ['Count']
        
        # Mark as present if detected more than once
        attendance['Present'] = 0
        for i in range(len(attendance)):
            if attendance['Count'][i] > 1:
                attendance['Present'][i] = 1

        attendance_final = attendance.drop(['Count'], axis=1)
        attendance_final.reset_index(inplace=True)
        attendance_final.rename(columns={'index': 'RegNo'}, inplace=True)

        # --- UPDATE OR APPEND TO CSV ---
        csv_path = os.path.join(curd, 'Attendance.csv')

        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
            if column_name not in df.columns:
                df[column_name] = 0
        else:
            df = pd.DataFrame(columns=['RegNo', 'Day1', 'Day2', 'Day3'])

        for idx, row in attendance_final.iterrows():
            regno = row['RegNo']
            present = row['Present']
            if regno in df['RegNo'].values:
                df.at[df.index[df['RegNo'] == regno][0], column_name] = present
            else:
                new_row = {'RegNo': regno, 'Day1': 0, 'Day2': 0, 'Day3': 0}
                new_row[column_name] = present
                df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)

        df.to_csv(csv_path, index=False)
        print(f"Attendance updated for {column_name}.")
        print(df)
        print("\nAttendance saved to Attendance.csv")
    else:
        print("No students detected for attendance.")

if __name__ == "__main__":
    take_attendance()
