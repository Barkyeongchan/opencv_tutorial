# landmark_utils.py

import numpy as np

def get_eye_landmarks(shape, eye_indices):
    # 눈 랜드마크 좌표 추출
    # eye_indices: 눈 영역 랜드마크 인덱스 리스트 (예: [36,37,38,39,40,41])
    return [(shape.part(i).x, shape.part(i).y) for i in eye_indices]

def calculate_ear(eye_landmarks):
    # EAR (Eye Aspect Ratio) 계산
    # eye_landmarks: 눈 좌표 리스트 6개 점
    def euclidean_dist(p1, p2):
        return np.linalg.norm(np.array(p1) - np.array(p2))

    # EAR 공식
    A = euclidean_dist(eye_landmarks[1], eye_landmarks[5])
    B = euclidean_dist(eye_landmarks[2], eye_landmarks[4])
    C = euclidean_dist(eye_landmarks[0], eye_landmarks[3])
    ear = (A + B) / (2.0 * C)
    return ear