# 모든 설정값을 한 곳에서 관리

# 랜드마크 모델 경로

LANDMARK_MODEL_PATH = './models/shape_predictor_68_face_landmarks.dat'

# EAR 관련 설정

EAR_THRESHOLD = 0.25

DROWSY_FRAMES_THRESHOLD = 20

# 눈 영역 랜드마크 인덱스

LEFT_EYE = [36, 37, 38, 39, 40, 41]

RIGHT_EYE = [42, 43, 44, 45, 46, 47]

# 화면 설정

WINDOW_WIDTH = 640

WINDOW_HEIGHT = 480

# 색상 설정 (BGR)

GREEN = (0, 255, 0)      # 정상

YELLOW = (0, 255, 255)   # 주의  

RED = (0, 0, 255)        # 위험
