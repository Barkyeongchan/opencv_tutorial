import cv2 
import matplotlib.pyplot as plt
import dlib
import tensorflow as tf 
import numpy as np

image = cv2.imread('../data/images/charles.jpg')

# 이미지 로딩 확인
if image is None:
    print("이미지를 로딩할 수 없습니다!")
    exit()

face_detector = dlib.get_frontal_face_detector()
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
face_detection = face_detector(gray)

# 디버깅 정보 출력
print(f"검출된 얼굴 수: {len(face_detection)}")
print(f"face_detection 타입: {type(face_detection)}")

# 얼굴 검출 결과 확인
if len(face_detection) == 0:
    print("얼굴이 검출되지 않았습니다.")
    exit()

# 첫 번째 얼굴 정보 가져오기
face_rect = face_detection[0]
print(f"face_rect 타입: {type(face_rect)}")

left, top, right, bottom = face_rect.left(), face_rect.top(), face_rect.right(), face_rect.bottom()
print(f"얼굴 좌표: left={left}, top={top}, right={right}, bottom={bottom}")

roi = image[top:bottom, left:right]
print(f"ROI shape: {roi.shape}")

# Resize image
roi = cv2.resize(roi, (48, 48))
print(f"Resized ROI shape: {roi.shape}")

# Normalize
roi = roi / 255

roi = np.expand_dims(roi, axis=0)
print(f"Final ROI shape: {roi.shape}")

network = tf.keras.models.load_model('../models/emotion_model.h5')

pred_probability = network.predict(roi)
print(pred_probability)

pred = np.argmax(pred_probability)
print(f"예측 결과: {pred}")