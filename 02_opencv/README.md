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
