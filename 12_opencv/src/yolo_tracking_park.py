from ultralytics import YOLO
import cv2

# YOLO 모델 로드
model = YOLO('yolo11n.pt')

# 비디오 열기
cap = cv2.VideoCapture('./video.mp4')

# 원본 FPS 가져오기
fps = cap.get(cv2.CAP_PROP_FPS)

start_seconds = 10  # 자를 앞부분 초
start_frame = int(fps * start_seconds)

print(f"FPS: {fps}, 자를 프레임 수: {start_frame}")

# 앞부분 프레임 버리기
for _ in range(start_frame):
    ret = cap.grab()  # 프레임을 읽고 버림
    if not ret:
        break

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