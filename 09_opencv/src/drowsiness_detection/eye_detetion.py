import cv2
import dlib
import numpy as np

# dlib 초기화
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('../shape_predictor_68_face_landmarks.dat')

# 카메라 열기
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, img = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)

    for rect in faces:
        shape = predictor(gray, rect)

        for eye_points in [(36, 41), (42, 47)]:  # 왼눈, 오른눈
            pts = []
            for i in range(eye_points[0], eye_points[1] + 1):
                x, y = shape.part(i).x, shape.part(i).y
                pts.append((x, y))
                cv2.circle(img, (x, y), 2, (0, 255, 0), -1)  # 눈 점 표시

            # 눈 바운딩 박스
            x_coords = [p[0] for p in pts]
            y_coords = [p[1] for p in pts]
            x_min, x_max = min(x_coords), max(x_coords)
            y_min, y_max = min(y_coords), max(y_coords)
            cv2.rectangle(img, (x_min, y_min), (x_max, y_max), (255, 0, 0), 1)

            # 눈 중심점 표시
            cx = int(np.mean(x_coords))
            cy = int(np.mean(y_coords))
            cv2.circle(img, (cx, cy), 3, (0, 0, 255), -1)

    cv2.imshow('Eye Detection', img)
    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()
