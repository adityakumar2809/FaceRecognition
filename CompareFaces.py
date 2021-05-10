import os
import sys
import cv2
import pickle
import numpy as np
import pandas as pd
import face_recognition
from sklearn.metrics import confusion_matrix


def getFaceDetectionPrediction(folder_path=None, facial_data_df=None):
    if folder_path is None or facial_data_df is None:
        print('Insufficient Data Specified!')
        sys.exit(0)
    y_pred = []
    for img_name in os.listdir(folder_path):
        img = face_recognition.load_image_file(f'{folder_path}/{img_name}')
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        face_encodings = face_recognition.face_encodings(img)
        if len(face_encodings) > 0:
            face_encodings = face_encodings[0]
            min_face_distance = 1
            min_face_label = 'invalid'
            for label, label_face_encoding in facial_data_df.items():
                face_distance =  face_recognition.face_distance(
                    [label_face_encoding],
                    face_encodings
                )
                if len(face_distance) > 0:
                    face_distance = face_distance[0]
                    if face_distance < min_face_distance:
                        min_face_distance = face_distance
                        min_face_label = label
            y_pred.append(min_face_label)
        else:
            y_pred.append('invalid')

    return y_pred


def main():
    test_folder = 'data/test'
    labels = os.listdir(test_folder)
    y_true = []
    y_pred = []
    facial_data_df = pickle.load(open('results/facial_data.pkl', 'rb'))
    for label in labels:
        folder_path = f'{test_folder}/{label}'
        y_pred += getFaceDetectionPrediction(folder_path, facial_data_df)
        y_true += [label for i in range(len(os.listdir(folder_path)))]

    cm = confusion_matrix(y_true, y_pred)
    print(cm)

if __name__ == '__main__':
    main()