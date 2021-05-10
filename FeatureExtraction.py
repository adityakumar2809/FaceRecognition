import os
import sys
import cv2
import pickle
import numpy as np
import pandas as pd
import face_recognition


def getEncodedFeatureValues(folder_path=None):
    if folder_path is None:
        print('No folder specified')
        sys.exit(0)
    face_encodings_list = []
    for img_name in os.listdir(folder_path):
        img = face_recognition.load_image_file(f'{folder_path}/{img_name}')
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        face_encodings = face_recognition.face_encodings(img)
        if len(face_encodings) > 0:
            face_encodings_list.append(face_encodings[0])
    face_encodings_array = np.asarray(face_encodings_list)
    mean_encoding = face_encodings_array.mean(axis=0)
    return mean_encoding


def main():
    train_folder = 'data/train'
    labels = os.listdir(train_folder)
    facial_data = {}
    for label in labels:
        mean_encoding = getEncodedFeatureValues(
            folder_path=f'{train_folder}/{label}'
        )
        facial_data[label] = mean_encoding
    facial_data_df = pd.DataFrame(facial_data)
    pickle.dump(facial_data_df, open('results/facial_data.pkl', 'wb'))

    print('Training Completed Successfully')


if __name__ == '__main__':
    main()