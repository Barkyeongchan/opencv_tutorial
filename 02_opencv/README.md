# openCV 색상 표현 방식와 쓰레시홀드(Threshold), 히스토그램(Histogram)

## 목차
1. 색상 표현 방식
2. 이미지 변환
3. 도형 그리기
4. 개인 프로젝트
   

## 1. 색상 표현 방식

1. **RGB와 BGR/BGRA**

RGB : Red, Green, Blue 순서대로 값을 표기함

BGR : openCV에서 사용하는 방식으로 RGB와 반대로 Blue, Green, Red 순서대로 값을 표기함

예) 빨강 = RGB에서는 (255, 0, 0) / BGR에서는 (0, 0, 255)로 표기함

BGRA : BGR에 A(alpha, 알파)가 추가된 표기법, 배경의 투명도가 추가된다.

```python3
import cv2
import numpy as np

# @이미지 기본 값으로 불러오기
img = cv2.imread('../img/opencv_logo.png')
  
# @이미지 BGR값으로 불러오기 / IMREAD_COLOR 옵션                   
bgr = cv2.imread('../img/opencv_logo.png', cv2.IMREAD_COLOR)

# @이미지 BGRA값으로 불러오기(알파 채널을 가진 경우) / # IMREAD_UNCHANGED 옵션
bgra = cv2.imread('../img/opencv_logo.png', cv2.IMREAD_UNCHANGED)

# 각 옵션에 따른 이미지 shape
print("default", img.shape, "color", bgr.shape, "unchanged", bgra.shape) 

# @이미지 출력
cv2.imshow('bgr', bgr)
cv2.imshow('bgra', bgra)
cv2.imshow('alpha', bgra[:,:,3])  # 알파 채널만 표시
cv2.waitKey(0)
cv2.destroyAllWindows()
```
<img width="536" height="226" alt="image" src="https://github.com/user-attachments/assets/022e5d6d-614a-4492-b7cf-1561252e2ad2" />

2. **회색조 이미지로 변환(Gray Scale)**

이미지 연산의 양을 줄여 연삭 속도를 높이는데 필요함

`cv2.imread(img, cv2.IMREAD_GRAYSCALE)` 함수를 사용한다.

```python3
import cv2
import numpy as np

img = cv2.imread('../img/yeosu.jpg')

img2 = img.astype(np.uint16)                # dtype 변경
b,g,r = cv2.split(img2)                     # 채널 별로 분리
#b,g,r = img2[:,:,0], img2[:,:,1], img2[:,:,2]
gray1 = ((b + g + r)/3).astype(np.uint8)    # 평균 값 연산후 dtype 변경

gray2 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # BGR을 그레이 스케일로 변경
cv2.imshow('original', img)
cv2.imshow('gray1', gray1)
cv2.imshow('gray2', gray2)

cv2.waitKey(0)
cv2.destroyAllWindows()
```
<img width="1280" height="359" alt="image" src="https://github.com/user-attachments/assets/727bda4b-d2b1-404f-b1d0-1cb10768b8df" />

회색조 뿐 아니라 다양한 색상 표현 방식으로 변환 할 수 있다.

`
cv2.COLOR_BGR2GRAY: BGR 색상 이미지를 회색조 이미지로 변환
cv2.COLOR_GRAY2BGR: 회색조 이미지를 BGR 색상 이미지로 변환
cv2.COLOR_BGR2RGB: BGR 색상 이미지를 RGB 색상 이미지로 변환
cv2.COLOR_BGR2HSV: BGR 색상 이미지를 HSV 색상 이미지로 변환
cv2.COLOR_HSV2BGR: HSV 색상 이미지를 BGR 색상 이미지로 변환
cv2.COLOR_BGR2YUV: BGR 색상 이미지를 YUV 색상 이미지로 변환
cv2.COLOR_YUV2BGR: YUB 색상 이미지를 BGR 색상 이미지로 변환
`

3. **HSV(Hue 색조, Saturation 채도, Value 명도) 방식**

<img width="640" height="480" alt="image" src="https://github.com/user-attachments/assets/42574139-0989-4e63-915d-38bc2af1775d" />

```python3
import cv2
import numpy as np

# @BGR 컬러 스페이스로 원색 픽셀 생성
red_bgr = np.array([[[0,0,255]]], dtype=np.uint8)   # 빨강 값만 갖는 픽셀
green_bgr = np.array([[[0,255,0]]], dtype=np.uint8) # 초록 값만 갖는 픽셀
blue_bgr = np.array([[[255,0,0]]], dtype=np.uint8)  # 파랑 값만 갖는 픽셀
yellow_bgr = np.array([[[0,255,255]]], dtype=np.uint8) # 노랑 값만 갖는 픽셀

# @BGR 컬러 스페이스를 HSV 컬러 스페이스로 변환
red_hsv = cv2.cvtColor(red_bgr, cv2.COLOR_BGR2HSV);
green_hsv = cv2.cvtColor(green_bgr, cv2.COLOR_BGR2HSV);
blue_hsv = cv2.cvtColor(blue_bgr, cv2.COLOR_BGR2HSV);
yellow_hsv = cv2.cvtColor(yellow_bgr, cv2.COLOR_BGR2HSV);

# @HSV로 변환한 픽셀 출력
print("red:",red_hsv)
print("green:", green_hsv)
print("blue", blue_hsv)
print("yellow", yellow_hsv)
```
