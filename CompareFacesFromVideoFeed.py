import sys
import cv2
import pickle
import numpy as np
import pandas as pd
import face_recognition


def getFaceDetectionPrediction(img):
    if facial_data_df is None:
        print('Insufficient Data Specified!')
        sys.exit(0)
    
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    face_locations = face_recognition.face_locations(img)
    face_encodings = face_recognition.face_encodings(img)

    y_pred = []
    
    if len(face_encodings) > 0:
        for face_index, face_encoding in enumerate(face_encodings):
            min_face_distance = 1
            min_face_label = 'invalid'
            for label, label_face_encoding in facial_data_df.items():
                face_distance =  face_recognition.face_distance(
                    [label_face_encoding],
                    face_encoding
                )
                face_compare_result =  face_recognition.compare_faces(
                    [label_face_encoding],
                    face_encoding
                )
                if len(face_distance) > 0:
                    face_distance = face_distance[0]
                    if not face_compare_result[0] == True:
                        continue
                    if face_distance < min_face_distance:
                        min_face_distance = face_distance
                        min_face_label = label
            y_pred.append((face_locations[face_index], min_face_label))

    return y_pred


def main():
    global facial_data_df
    facial_data_df = pickle.load(open('results/facial_data.pkl', 'rb'))

    cap = cv2.VideoCapture('data/livefeed/aditya.mp4')
    out = None

    while True:
        ret, img = cap.read()
        if not ret:
            break

        if not out:
            out = cv2.VideoWriter(
                'results/livefeed/aditya.avi',
                cv2.VideoWriter_fourcc(*'DIVX'),
                cap.get(cv2.CAP_PROP_FPS), 
                (img.shape[1], img.shape[0])
            )

        faces = getFaceDetectionPrediction(img)
        
        for face_locations, face_label in faces:
            cv2.rectangle(
                img,
                (face_locations[3], face_locations[0]),
                (face_locations[1], face_locations[2]),
                (0, 255, 0),
                2
            )
            cv2.putText(
                img,
                face_label,
                (face_locations[3], face_locations[0]),
                cv2.FONT_HERSHEY_COMPLEX,
                1,
                (0, 255, 0),
                2
            )



        # cv2.imshow('img', img)
        out.write(img)
        k = cv2.waitKey(1) & 0xff
        if k == 27:
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()