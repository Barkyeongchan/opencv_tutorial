# 힉습 목표 : YOLO를 활용하여 컴퓨터 비전 프로젝트를 진행한다.

# YOLO

## 목차

1. YOLO
   - YOLO란?
   - YOLO 모델별 특징
   - YOLO11 설치
   - image 예제 실습
   - 카메라 캡처 예제 실습
   - 신뢰도(confidence)조절 예제 실습
   - 실시간 교통상황 감지 예제 실습
  
2. 개인 프로젝트 (원하는 객체만 탐지)

## 1. YOLO

<details>
<summary></summary>
<div markdown="1">

## **1-1. YOLO란?**

 _You Only Look Once_ 의 약자로, **객체 탐지(Object Detection) 분야에서 많이 쓰이는 딥러닝 모델**

 한 번에 전체 이미지를 보고 **객체의 위치와 종류를 동시에 예측**하는 방식

| 특징 | 설명 |
|------|------|
| 속도 | 한 번의 신경망 연산으로 모든 객체를 감지하므로 매우 빠름 |
| End-to-End 구조 | 이미지 입력 → 바로 박스 좌표와 클래스 출력 |
| 실시간 처리 가능 | 적당한 하드웨어면 웹캠·CCTV 등 실시간 영상 처리 가능 |

**[작동 방식]**

**1-Stage Detection** : 영역 추정(region proposal)"과 "분류(classification)"를 한 번에 처리

전체 이미지를 그리드(grid)로 나눠서 각 셀(cell)마다 객체를 탐지

<img width="934" height="877" alt="image" src="https://github.com/user-attachments/assets/6a52c0c5-c944-4ef7-96d7-f364b7b10a98" />

## **1-2. YOLO 모델별 특징**

| 모델         | 특징 요약 |
|--------------|----------|
| YOLO v1      | 2-stage → 1-stage 전환의 선구자, 속도 빠름, 정확도 낮음 |
| SSD          | 속도·성능 모두 우수 |
| YOLO v2      | SSD와 성능 비슷, SSD보다 빠름, 작은 물체 탐지 성능 낮음 |
| Retinanet    | YOLO v2보다 느리지만 성능 우수, FPN 적용 |
| YOLO v3      | YOLO v2보다 속도 약간 느림, 성능 크게 향상, FPN 포함 |
| EfficientDet | D0~D7 모델, D0는 YOLO v3보다 빠르고 성능도 약간 우수 |

## **1-3. YOLO11 설치**

**[1. YOLO 학습용 가상환경 생성]**

```trminal
python -m venv yolovenv
```

_.gitignore에 yolovenv가상환경 추가_

**[2. YOLO11 다운로드]**

[Ultralytics YOLO11](https://docs.ultralytics.com/ko/models/yolo11/)

```terminal
pip install ultralytics
```

<img width="192" height="74" alt="image" src="https://github.com/user-attachments/assets/50efee40-1e77-41e0-b62c-1784b0ef5ef7" />

## **1-4. image 예제 실습**

**[1. yolo_image.py 생성후 실행]**

```python3
from ultralytics import YOLO

model = YOLO('yolo11n.pt')  # YOLO 버전 지정
```

실행 후 `yolo11n.pt`가 자동 다운로드 됨

<br><br>

**[2. 에제 이미지 다운로드]**

```python3
from ultralytics import YOLO

model = YOLO('yolo11n.pt')

results = model('http://ultralytics.com/images/bus.jpg')

results[0].show()
```

코드를 실행하면 아래와 같은 이미지 출력

<img width="511" height="680" alt="image" src="https://github.com/user-attachments/assets/e4ae4c5d-36d8-4320-8e7b-c01b8b551f8c" />

**[3. 객체 검출 코드 추가]**

```python3
from ultralytics import YOLO

# YOLO 모델 설정
model = YOLO('yolo11n.pt')

# results = model('http://ultralytics.com/images/bus.jpg')

test_images = [
    'https://ultralytics.com/images/zidan.jpg'
    'https://ultralytics.com/images/bus.jpg'
]

for img in test_images:
    results = model(img)
    print(f'검출된 객체 수 : {len(results[0].boxes)}')

results[0].show()
```

`bus.jpg` 이미지에 대한 결과 값 출력

<img width="135" height="22" alt="image" src="https://github.com/user-attachments/assets/cac85cee-f30e-4da9-828b-9099f4f1c730" />

## **1-5. 카메라 캡처 예제 실습**

**[1. 코드 생성]**

```python3
from ultralytics import YOLO
import cv2

model = YOLO('yolo11n.pt')

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if ret:
        results = model(frame, verbose= False)
        annotated_frame = results[0].plot()
        cv2.imshow('Yolo ', annotated_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
```

**[2. 결과 출력]**

<img width="638" height="509" alt="image" src="https://github.com/user-attachments/assets/3171e6ba-e6f7-4856-896c-dadcbb9e88d2" />

## **1-6. 신뢰도(confidence)조절 예제 실습**

**[1. 코드 생성]**

```python3
from ultralytics import YOLO
import cv2  

# YOLO 모델 로드
model = YOLO('yolo11n.pt')

# 신뢰도별 검출 결과 비교
confidence_levels = [0.25, 0.5, 0.75]
test_image = 'https://ultralytics.com/images/bus.jpg'

print("🧪 신뢰도별 검출 실험:")
for conf in confidence_levels:
    results = model(test_image, conf=conf, verbose=False)
    num_objects = len(results[0].boxes) if results[0].boxes else 0
    print(f"신뢰도 {conf}: {num_objects}개 객체 검출")

# 실시간 신뢰도 조정 도구
confidence = 0.5
cap = cv2.VideoCapture(0)

print("키보드 조작: +/- 로 신뢰도 조정, q로 종료")
while True:
    ret, frame = cap.read()
    if ret:
        results = model(frame, conf=confidence, verbose=False)
        annotated = results[0].plot()
        
        # 현재 설정 표시
        info_text = f"Confidence: {confidence:.2f}"
        cv2.putText(annotated, info_text, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        cv2.imshow('Confidence Tuner', annotated)
    
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('+') or key == ord('='):
        confidence = min(0.95, confidence + 0.1)
    elif key == ord('-'):
        confidence = max(0.1, confidence - 0.1)
```

**[2. 결과 출력]**

<img width="637" height="507" alt="image" src="https://github.com/user-attachments/assets/7bec2c41-c5df-40e1-aa51-4d56143c978c" />

<img width="639" height="507" alt="image" src="https://github.com/user-attachments/assets/f55115d7-a472-4f4e-882b-a639ba85b321" />

## **1-7. 실시간 교통상황 감지 예제 실습**

**[1. 코드 생성]**

```python3
from ultralytics import YOLO
import cv2  

class TrafficMonitor:
    def __init__(self):
        self.model = YOLO('yolo11n.pt')
        self.traffic_classes = {
            0: 'person', 2: 'car', 3: 'motorcycle', 
            5: 'bus', 7: 'truck', 9: 'traffic_light'
        }
        self.stats = {'total_detections': 0, 'by_class': {}}
    
    def analyze_frame(self, frame):
        results = self.model(frame, 
                           classes=list(self.traffic_classes.keys()), 
                           conf=0.5, verbose=False)
        
        frame_stats = {'vehicles': 0, 'pedestrians': 0, 'signals': 0}
        
        if results[0].boxes is not None:
            for box in results[0].boxes:
                class_id = int(box.cls[0])
                class_name = self.traffic_classes[class_id]
                
                if class_id in [2, 3, 5, 7]:  # vehicles
                    frame_stats['vehicles'] += 1
                elif class_id == 0:  # person
                    frame_stats['pedestrians'] += 1
                elif class_id == 9:  # traffic_light
                    frame_stats['signals'] += 1
                
                self.stats['by_class'][class_name] = \
                    self.stats['by_class'].get(class_name, 0) + 1
            
            self.stats['total_detections'] += len(results[0].boxes)
        
        return results[0].plot(), frame_stats
    
    def run_live_monitoring(self):
        cap = cv2.VideoCapture(0)
        print("🚗 교통 모니터링 시작! q로 종료")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            annotated_frame, frame_stats = self.analyze_frame(frame)
            
            # 정보 표시
            y = 30
            cv2.putText(annotated_frame, f"Vehicles: {frame_stats['vehicles']}", 
                       (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            y += 30
            cv2.putText(annotated_frame, f"Pedestrians: {frame_stats['pedestrians']}", 
                       (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            y += 30
            cv2.putText(annotated_frame, f"Total Detected: {self.stats['total_detections']}", 
                       (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
            
            cv2.imshow('Traffic Monitoring System', annotated_frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()
        self.show_final_stats()
    
    def show_final_stats(self):
        print("\n📊 최종 통계:")
        print(f"총 검출 횟수: {self.stats['total_detections']}")
        print("클래스별 검출 현황:")
        for class_name, count in self.stats['by_class'].items():
            print(f"  {class_name}: {count}회")

# 시스템 실행
monitor = TrafficMonitor()
monitor.run_live_monitoring()
```

**[2. 결과 출력]**

<img width="638" height="508" alt="image" src="https://github.com/user-attachments/assets/bad99a11-25db-44d7-98d5-a8fea946da05" />

<img width="636" height="507" alt="image" src="https://github.com/user-attachments/assets/46211b46-f3e3-4f8d-8811-33cddbe24050" />

<img width="161" height="144" alt="image" src="https://github.com/user-attachments/assets/dfb4a25b-c1b0-41c9-b67d-b8f534757057" />

</div>
</details>

## 2. 개인 프로젝트 (원하는 객체만 탐지)

<details>
<summary></summary>
<div markdown="1">

## **2-1. 완성 코드**

```python3
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
```

## **2-2. 결과 이미지**

**[1. 영상 초기 객체 인식]**

<img width="358" height="667" alt="image" src="https://github.com/user-attachments/assets/b52e4c88-197d-4221-954a-43f0b374c54f" />

<br><br>

**[2. 필요 없는 객체 제거]**

<img width="360" height="671" alt="image" src="https://github.com/user-attachments/assets/6491824a-c14e-4712-b3e8-ea273803d87b" />

<br><br>

**[3. 객체 이름 변경 (motorcycle, bicycle → unknown)]**

<img width="368" height="671" alt="image" src="https://github.com/user-attachments/assets/9a758e30-6336-4111-9f57-f5317323a122" />

<br><br>

**[4. 객체 이름 변경 (chair → person)]**

<img width="358" height="668" alt="image" src="https://github.com/user-attachments/assets/c9e7d77a-ea91-4013-b9e4-8063861e4c88" />

<br><br>

**[5. 이후 영상 출력]**

<img width="355" height="668" alt="image" src="https://github.com/user-attachments/assets/ca56efdd-50d1-4cb6-a73a-78a5c291a715" />

<img width="358" height="668" alt="image" src="https://github.com/user-attachments/assets/bf39f701-0191-493b-92d8-9327b6b0719d" />

</div>
</details>
