import cv2
import numpy as np
import face_recognition


img = face_recognition.load_image_file('data/MukeshAmbaniImage1.jpg')
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
test_img = face_recognition.load_image_file('data/MukeshAmbaniImage2.jpg')
test_img = cv2.cvtColor(test_img, cv2.COLOR_BGR2RGB)

cv2.imshow('Mukesh Ambani - Train', img)
cv2.waitKey(0)
