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

excluded_classes = [0, 72]  # person, refrigerator 제외
all_classes = list(range(80))
included_classes = [c for c in all_classes if c not in excluded_classes]

# 이름을 바꿀 클래스 번호와 대응 이름 (motorcycle, bicycle → unknown, chair → person)
rename_map = {
    3: "unknown",  # motorcycle
    1: "unknown",  # bicycle
    56: "person"   # chair
}

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # YOLO로 객체 탐지 (제외 클래스 제외)
    results = model(frame, classes=included_classes)

    # 원본 이미지 복사
    img = frame.copy()

    # 탐지된 박스 좌표, 클래스 번호, 신뢰도 가져오기
    boxes = results[0].boxes.xyxy.cpu().numpy().astype(int)  # (N, 4)
    classes = results[0].boxes.cls.cpu().numpy().astype(int)  # (N,)
    scores = results[0].boxes.conf.cpu().numpy()             # (N,)

    # 탐지된 객체 하나씩 반복
    for box, cls, score in zip(boxes, classes, scores):
        x1, y1, x2, y2 = box
        # 기본 클래스명 가져오기
        cls_name = model.names[cls]

        # rename_map에 있으면 이름 바꾸기, 없으면 기본 이름 사용
        display_name = rename_map.get(cls, cls_name)

        label = f"{display_name} {score:.2f}"

        # 박스와 텍스트 그리기
        color = (0, 255, 0)  # 초록색
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        cv2.putText(img, label, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    # 화면 표시
    cv2.imshow("YOLO Detection", img)

    # q 키 누르면 종료
    if cv2.waitKey(delay) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()