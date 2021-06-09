import os
import sys
import cv2
import pickle
import numpy as np
import pandas as pd
import face_recognition
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix


def getFaceDetectionPrediction(folder_path=None, facial_data_df=None):
    if folder_path is None or facial_data_df is None:
        print('Insufficient Data Specified!')
        sys.exit(0)
    y_pred = []
    for img_counter, img_name in enumerate(os.listdir(folder_path)):
        img = face_recognition.load_image_file(f'{folder_path}/{img_name}')
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        face_encodings = face_recognition.face_encodings(img)
        face_locations = face_recognition.face_locations(img)
        if len(face_encodings) > 0:
            face_encodings = face_encodings[0]
            face_locations = face_locations[0]
            min_face_distance = 1
            min_face_label = 'invalid'
            for label, label_face_encoding in facial_data_df.items():
                face_distance =  face_recognition.face_distance(
                    [label_face_encoding],
                    face_encodings
                )
                face_compare_result =  face_recognition.compare_faces(
                    [label_face_encoding],
                    face_encodings
                )
                if len(face_distance) > 0:
                    face_distance = face_distance[0]
                    if not face_compare_result[0] == True:
                        continue
                    if face_distance < min_face_distance:
                        min_face_distance = face_distance
                        min_face_label = label
            y_pred.append(min_face_label)
            cv2.rectangle(
                img,
                (face_locations[3], face_locations[0]),
                (face_locations[1], face_locations[2]),
                (0, 255, 0),
                2
            )
            cv2.putText(
                img,
                min_face_label,
                (face_locations[3], face_locations[0]),
                cv2.FONT_HERSHEY_COMPLEX,
                1,
                (0, 255, 0),
                2
            )
            cv2.imwrite(f'results/images/{min_face_label}_{img_counter}.jpg', img)
        else:
            y_pred.append('invalid')

    return y_pred


def plotConfusionMatrix(labels, y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred, labels)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(cm)
    plt.title('Confusion Matrix')
    fig.colorbar(cax)
    ax.set_xticklabels([''] + labels)
    ax.set_yticklabels([''] + labels)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()


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

    plotConfusionMatrix(labels, y_true, y_pred)


if __name__ == '__main__':
    main()