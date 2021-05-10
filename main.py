import cv2
import numpy as np
import face_recognition


img = face_recognition.load_image_file('data/MukeshAmbaniImage1.jpg')
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
test_img = face_recognition.load_image_file('data/MukeshAmbaniImage2.jpg')
test_img = cv2.cvtColor(test_img, cv2.COLOR_BGR2RGB)

face_locations = face_recognition.face_locations(img)[0]
encode_locations = face_recognition.face_encodings(img)[0]

cv2.rectangle(
    img,
    (face_locations[3], face_locations[0]),
    (face_locations[1], face_locations[2]),
    (0, 255, 0),
    2
)

cv2.imshow('Mukesh Ambani - Train', img)
cv2.waitKey(0)
