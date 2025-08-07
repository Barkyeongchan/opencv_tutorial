import cv2
import dlib
from utils.landmark_utils import get_eye_landmarks, calculate_ear

# 설정값은 settings.py에서 가져온다고 가정
import config.settings as settings

# 상태 변수
consecutive_frames = 0

def detect_faces(frame, detector):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)
    return faces

def get_landmarks(frame, face_rect, predictor):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    shape = predictor(gray, face_rect)
    return shape

def calculate_ear_from_landmarks(landmarks):
    left_eye = get_eye_landmarks(landmarks, settings.LEFT_EYE)
    right_eye = get_eye_landmarks(landmarks, settings.RIGHT_EYE)
    left_ear = calculate_ear(left_eye)
    right_ear = calculate_ear(right_eye)
    return (left_ear + right_ear) / 2.0

def check_drowsiness(ear_value):
    global consecutive_frames
    if ear_value < settings.EAR_THRESHOLD:
        consecutive_frames += 1
    else:
        consecutive_frames = 0

    if consecutive_frames == 0:
        return 'NORMAL'
    elif consecutive_frames < 10:
        return 'DROWSY'
    elif consecutive_frames < 20:
        return 'ALERT'
    else:
        return 'DANGER'

def draw_results(frame, landmarks, ear_value, drowsiness_level):
    left_eye = get_eye_landmarks(landmarks, settings.LEFT_EYE)
    right_eye = get_eye_landmarks(landmarks, settings.RIGHT_EYE)

    for (x, y) in left_eye + right_eye:
        cv2.circle(frame, (x, y), 2, settings.GREEN, -1)

    cv2.putText(frame, f"EAR: {ear_value:.3f}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, settings.GREEN, 2)

    color_map = {
        'NORMAL': settings.GREEN,
        'DROWSY': settings.YELLOW,
        'ALERT': (0, 165, 255),  # 주황색
        'DANGER': settings.RED
    }
    color = color_map.get(drowsiness_level, settings.GREEN)
    cv2.putText(frame, drowsiness_level, (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 3)

def main():
    global consecutive_frames
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(settings.LANDMARK_MODEL_PATH)
    cap = cv2.VideoCapture(0)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        faces = detect_faces(frame, detector)
        if len(faces) > 0:
            landmarks = get_landmarks(frame, faces[0], predictor)
            ear_value = calculate_ear_from_landmarks(landmarks)
            drowsiness_level = check_drowsiness(ear_value)
            draw_results(frame, landmarks, ear_value, drowsiness_level)
        else:
            consecutive_frames = 0
            cv2.putText(frame, "No face detected", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, settings.RED, 2)

        cv2.imshow('Drowsiness Detection', frame)
        if cv2.waitKey(1) == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
