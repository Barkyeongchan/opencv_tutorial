import cv2
import dlib
import numpy as np

# EAR 계산 함수
def calculate_ear(eye_points):
    # eye_points는 눈의 6개 랜드마크 좌표 리스트
    A = np.linalg.norm(np.array(eye_points[1]) - np.array(eye_points[5]))
    B = np.linalg.norm(np.array(eye_points[2]) - np.array(eye_points[4]))
    C = np.linalg.norm(np.array(eye_points[0]) - np.array(eye_points[3]))
    ear = (A + B) / (2.0 * C)
    return ear

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

        ears = []
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

            # EAR 계산
            ear = calculate_ear(pts)
            ears.append(ear)

        # 양 눈 EAR 평균
        avg_ear = np.mean(ears)

        # EAR 값을 이미지에 표시
        cv2.putText(img, f'EAR: {avg_ear:.3f}', (30, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    cv2.imshow('Eye Detection + EAR', img)
    if cv2.waitKey(1) == 27:  # ESC
        break

cap.release()
cv2.destroyAllWindows()