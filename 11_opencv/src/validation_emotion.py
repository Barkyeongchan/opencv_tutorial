import cv2
import matplotlib.pyplot as plt
import dlib
import tensorflow as tf
import numpy as np


image = cv2.imread('../data/img/charles.jpg')

# plt.figure(figsize=(8,8))
# plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
# plt.show()

face_detector = dlib.cnn_face_detection_model_v1('../data/weights/mmod_human_face_detector.dat')
face_detection = face_detector(image, 1)

left, top, right, bottom = face_detection[0].rect.left(),\
      face_detection[0].rect.top(), face_detection[0].rect.right(), face_detection[0].rect.bottom()
roi = image[top:bottom, left:right]

# cv2.imshow('roi', roi)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

print(roi.shape)

# 이미지 리사이징
roi = cv2.resize(roi, (48, 48))
print(roi.shape)

# 정규화
roi = roi / 255

roi = np.expand_dims(roi, axis=0)
print(roi.shape)

network = tf.keras.model.load_model('../data/models/emotion_model.h5')

pred_probability = network.predict(roi)
print(pred_probability)

pred = np.argmax(pred_probability)
print(pred)

print(test_dataset.class_indices)