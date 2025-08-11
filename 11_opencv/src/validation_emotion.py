import cv2
import matplotlib.pyplot as plt

image = cv2.imread('../data/img/charles.jpg')

# plt.figure(figsize=(8,8))
# plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
# plt.show()

face_detector = dlib.cnn_face_detection_model_v1('../data/weights/mmod_human_face_detector.dat')
face_detection = face_detector(image, 1)

