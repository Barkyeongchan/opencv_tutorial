from ultralytics import YOLO
import cv2

# YOLO 모델 로드
model = YOLO('yolo11n.pt')

# 비디오 열기
cap = cv2.VideoCapture('./video.mp4')

# 원본 FPS 가져오기
fps = cap.get(cv2.CAP_PROP_FPS)

# 2배 속도 → 대기 시간 절반
delay = max(1, int(1000 / (fps * 2)))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # YOLO로 객체 탐지
    results = model(frame)

    # 시각화
    annotated_frame = results[0].plot()

    # 화면 표시
    cv2.imshow("YOLO Detection", annotated_frame)

    # q 키 누르면 종료
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()