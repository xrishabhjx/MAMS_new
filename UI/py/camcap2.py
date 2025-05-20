#!/usr/bin/env python
# coding: utf-8

import math
import cv2
import time
from sklearn import neighbors
import numpy as np
import pandas as pd
import os
import os.path
import pickle
from PIL import Image, ImageDraw
import face_recognition
import sys

# --- CONFIG ---
curd = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(curd, "assets", "models", "trained_knn_model.clf")
ATTENDANCE_CSV = os.path.join(curd, "UI", "Attendance.csv")
TEST_IMAGES_DIR = os.path.join(curd, "assets", "test")
HISTORY_DIR = os.path.join(curd, "assets", "History")
DAY = "Day1"  # Default day; can be set via sys.argv

if len(sys.argv) > 1:
    DAY = sys.argv[1]  # e.g., "Day1", "Day2", "Day3"

# --- CLEANUP TEST IMAGES ---
import shutil
os.makedirs(TEST_IMAGES_DIR, exist_ok=True)
os.makedirs(HISTORY_DIR, exist_ok=True)
for i in os.listdir(TEST_IMAGES_DIR):
    src = os.path.join(TEST_IMAGES_DIR, i)
    dst = os.path.join(HISTORY_DIR, i)
    if os.path.exists(dst):
        os.remove(dst)  # Remove the existing file
    shutil.move(src, HISTORY_DIR)

# --- CAPTURE IMAGES ---
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FPS, 30)
count = 0
while count < 10:
    ret, frame = cap.read()
    if not ret:
        print("Failed to capture frame")
        continue
    curtime = time.strftime("%Y_%m_%d-%H_%M_%S")
    img_path = os.path.join(TEST_IMAGES_DIR, f"{curtime}.jpg")
    cv2.imwrite(img_path, frame)
    count += 1
    cv2.imshow('img', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    time.sleep(1)
cap.release()
cv2.destroyAllWindows()

# --- LOAD MODEL ---
def predict(frame, knn_clf=None, model_path=None, distance_threshold=0.6):
    if knn_clf is None and model_path is None:
        raise Exception("Must supply knn classifier either through knn_clf or model_path")
    if knn_clf is None:
        with open(model_path, 'rb') as f:
            knn_clf = pickle.load(f)
    X_img = frame
    X_face_locations = face_recognition.face_locations(X_img)
    if len(X_face_locations) == 0:
        return []
    faces_encodings = face_recognition.face_encodings(X_img, known_face_locations=X_face_locations)
    closest_distances = knn_clf.kneighbors(faces_encodings, n_neighbors=1)
    are_matches = [closest_distances[0][i][0] <= distance_threshold for i in range(len(X_face_locations))]
    return [(pred, loc) if rec else ("unknown", loc) for pred, loc, rec in zip(knn_clf.predict(faces_encodings), X_face_locations, are_matches)]

# --- RECOGNIZE FACES ---
names = []
for image_file in os.listdir(TEST_IMAGES_DIR):
    full_file_path = os.path.join(TEST_IMAGES_DIR, image_file)
    print("Looking for faces in {}".format(image_file))
    frame = cv2.imread(full_file_path, -1)
    if frame is None:
        continue
    predictions = predict(frame, model_path=MODEL_PATH)
    for name, (top, right, bottom, left) in predictions:
        print("- Found {} at ({}, {})".format(name, left, top))
        names.append(name)
    # Optionally show image with labels
    # final_img = show_prediction_labels_on_image(frame, predictions)
    # cv2.imshow("X", final_img)
    # cv2.waitKey(1)
# cv2.destroyAllWindows()

# --- LOG ATTENDANCE ---
namesD = pd.DataFrame(names, columns=["Names"])
namesD = namesD[namesD.Names != "unknown"]
if namesD.empty:
    print("No known faces detected.")
    sys.exit(0)

attendance = pd.DataFrame(namesD.iloc[:, 0].value_counts())
attendance.rename(index=str, columns={'Names': 'Count'}, inplace=True)
attendance["Present"] = 0
for i in range(attendance.shape[0]):
    if attendance["Count"].iloc[i] > 5:
        attendance["Present"].iloc[i] = 1

attendance_final = attendance.drop(['Count'], axis=1)
attendance_final.reset_index(inplace=True)
attendance_final.rename(columns={'index': 'RegNo'}, inplace=True)

# --- UPDATE OR APPEND TO CSV ---
if os.path.exists(ATTENDANCE_CSV):
    df = pd.read_csv(ATTENDANCE_CSV)
    if DAY not in df.columns:
        df[DAY] = 0
else:
    df = pd.DataFrame(columns=['RegNo', 'Day1', 'Day2', 'Day3'])

for idx, row in attendance_final.iterrows():
    regno = row['RegNo']
    present = row['Present']
    if regno in df['RegNo'].values:
        df.at[df.index[df['RegNo'] == regno][0], DAY] = present
    else:
        new_row = {'RegNo': regno, 'Day1': 0, 'Day2': 0, 'Day3': 0}
        new_row[DAY] = present
        df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)

df.to_csv(ATTENDANCE_CSV, index=False)
print(f"Attendance updated for {DAY}.")
print(df)




