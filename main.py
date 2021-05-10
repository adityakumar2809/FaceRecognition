import cv2
import numpy as np
import face_recognition


img = face_recognition.load_image_file('data/MukeshAmbaniImage1.jpg')
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img_test = face_recognition.load_image_file('data/RatanTataImage1.jpg')
img_test = cv2.cvtColor(img_test, cv2.COLOR_BGR2RGB)

face_locations = face_recognition.face_locations(img)[0]
encode_locations = face_recognition.face_encodings(img)[0]

cv2.rectangle(
    img,
    (face_locations[3], face_locations[0]),
    (face_locations[1], face_locations[2]),
    (0, 255, 0),
    2
)

face_locations_test = face_recognition.face_locations(img_test)[0]
encode_locations_test = face_recognition.face_encodings(img_test)[0]

cv2.rectangle(
    img_test,
    (face_locations_test[3], face_locations_test[0]),
    (face_locations_test[1], face_locations_test[2]),
    (0, 255, 0),
    2
)


results = face_recognition.compare_faces([encode_locations], encode_locations_test)
print(results)

cv2.imshow('Mukesh Ambani - Train', img)
cv2.imshow('Mukesh Ambani - Test', img_test)
cv2.waitKey(0)
