# 힉습 목표 : YOLO를 활용하여 컴퓨터 비전 프로젝트를 진행한다.

# YOLO

## 목차

1. YOLO
   - YOLO란?
   - YOLO 모델별 특징
   - YOLO11 설치
   - 예제 실습

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

## **1-4. 예제 실습**

**[1. helloworld.py 생성후 실행]**

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


</div>
</details>

