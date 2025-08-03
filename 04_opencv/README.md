# 이미지 제어와 이미지 뒤틀기 / 영상 필터와 블러링 / 경계 검출

## 목차
1. 이미지 제어
2. 이미지 뒤틀기
3. 개인 프로젝트 (자동차 번호판 추출)
4. 영상 필터와 컨볼루션
5. 블러링
6. 경계 검출

## 1. 이미지 제어
<details>
<summary></summary>
<div markdown="1">

1. **이미지 이동(Translation)**

**이미지 이동이란?**

원래 있던 좌표에 이동하려는 거리만큼 더하여 이미지를 이동시키는 방법

```
x_new = x_old + d₁
y_new = y_old + d₂
```

<img width="354" height="105" alt="image" src="https://github.com/user-attachments/assets/86735ca7-f85c-4534-aa56-3f0ea7407b40" />

cv2.warpAffine 함수를 사용한다.

```
dst = cv2.warpAffine(src, matrix, dsize, dst, flags, borderMode, borderValue)
```

`src` : 원본 이미지, numpy 배열
`matrix` : 2 x 3 변환행렬, dtype=float32
`dsize` : 결과 이미지의 크기, (width, height)
`flags(optional)` : 보간법 알고리즘 플래그
`borderMode(optional)` : 외곽 영역 보정 플래그
`borderValue(optional)` : cv2.BORDER_CONSTANT 외곽 영역 보정 플래그일 경우 사용할 색상 값 (default=0)
`dst` : 결과 이미지


_flags 값_
`cv2.INTER_LINEAR` default 값, 인접한 4개 픽셀 값에 거리 가중치 사용
`cv2.INTER_NEAREST` 가장 가까운 픽셀 값 사용
`cv2.INTER_AREA` 픽셀 영역 관계를 이용한 재샘플링
`cv2.INTER_CUBIC` 인정합 16개 픽셀 값에 거리 가중치 사용

_borderMode 값_
`cv2.BORDER_CONSTANT` 고정 색상 값
`cv2.BORDER_REPLICATE` 가장자리 복제
`cv2.BORDER_WRAP` 반복
`cv2.BORDER_REFLECT` 반사

```python3
# 평행 이동

import cv2
import numpy as np


img = cv2.imread('../img/fish.jpg')
rows, cols = img.shape[0:2]  # 영상의 크기 정의

dx, dy = 100, 50            # 이동할 픽셀 거리 정의

# @변환 행렬 생성
mtrx = np.float32([[1, 0, dx], [0, 1, dy]])

# @단순 이동
dst = cv2.warpAffine(img, mtrx, (cols+dx,rows+dy))

# @탈락된 외곽 픽셀을 파랑색으로 보정
dst2 = cv2.warpAffine(img, mtrx, (cols+dx, rows+dy), None, \
                      cv2.INTER_LINEAR, cv2. BORDER_CONSTANT, (255, 0, 0))

# @탈락된 외곽 필섹을 원본으로 반사시켜서 보정
dst3 = cv2.warpAffine(img, mtrx, (cols+dx, rows+dy), None, \
                      cv2.INTER_LINEAR, cv2.BORDER_REFLECT)

# @이미지 출력
cv2.imshow('original', img)
cv2.imshow('trans',dst)
cv2.imshow('BORDER_CONSTATNT', dst2)
cv2.imshow('BORDER_FEFLECT', dst3)

cv2.waitKey(0)
cv2.destroyAllWindows()
```

<img width="1073" height="1027" alt="image" src="https://github.com/user-attachments/assets/06e0185c-a725-4bc3-9f5e-2dfcb663055e" />


2. **이미지 확대/축소(Scaling)**

**이미지 확대/축소란?**

원래 있던 좌표에 이동 하려는 거리만큼 곱한다

```
x_new = a₁ * x_old
y_new = a₂ * y_old
```

<img width="321" height="95" alt="image" src="https://github.com/user-attachments/assets/b3afdf9d-d392-4e4e-8f8b-5ca55578822f" />

cv2.resize() 함수를 사용한다.

```
cv2.resize(src, dsize, dst, fx, fy, interpolation)\
```

`src` : 입력 원본 이미지
`dsize` : 출력 영상 크기(확대/축소 목표 크기, (width, height)형식), 생략하면 fx, fy 배율을 적용
`fx, fy` : 크기 배율, dsize가 주어지면 dsize를 적용함
`interpolation` : 보간법 알고리즘 선택 플래그 (cv2.warpAffine()과 동일)
`dst` : 결과 이미지

> _보간법 (Interpolation)_
> 알려진 몇 개의 데이터 점을 바탕으로, 그 사이 존재하는 값을 추정하는 방법

```python3
# 이미지 확대, 축소

import cv2
import numpy as np

img = cv2.imread('../img/fish.jpg')
height, width = img.shape[:2]   # 영상 크기 정의

# @0.5배 축소 변환 행렬
m_small = np.float32([[0.5, 0, 0],
                       [0, 0.5,0]])  
# @2배 확대 변환 행렬
m_big = np.float32([[2, 0, 0],
                     [0, 2, 0]])  

# @보간법 적용 없이 확대 축소
dst1 = cv2.warpAffine(img, m_small, (int(height*0.5), int(width*0.5)))
dst2 = cv2.warpAffine(img, m_big, (int(height*2), int(width*2)))

# @보간법 적용한 확대 축소
dst3 = cv2.warpAffine(img, m_small, (int(height*0.5), int(width*0.5)), \
                        None, cv2.INTER_AREA)
dst4 = cv2.warpAffine(img, m_big, (int(height*2), int(width*2)), \
                        None, cv2.INTER_CUBIC)

# @cv.2resize() 함수를 사용해 확대 축소
# 크기 지정으로 축소
func1 = cv2.resize(img, (int(width*0.5), int(height*0.5)), \
                         interpolation=cv2.INTER_AREA)

# 배율 지정으로 확대
func2 = cv2.resize(img, None,  None, 2, 2, cv2.INTER_CUBIC)

# @이미지 출력
cv2.imshow("original", img)
cv2.imshow("small", dst1)
cv2.imshow("big", dst2)
cv2.imshow("small INTER_AREA", dst3)
cv2.imshow("big INTER_CUBIC", dst4)
cv2.imshow("use Function small", func1)
cv2.imshow("use Function big", func2)

cv2.waitKey(0)
cv2.destroyAllWindows()
```

<img width="766" height="815" alt="image" src="https://github.com/user-attachments/assets/4a6f2459-82e3-41ec-bd70-8c554da0ef03" />


3. **이미지 회전(Rotation)**

**이미지 회전을 위한 변환 행렬식**

<img width="966" height="638" alt="image" src="https://github.com/user-attachments/assets/82fd1e5b-35de-4ac7-956c-77c271e679e4" />


> _호도법_
> 원의 반지름과 호의 길이의 비율을 이용해 각도를 나타내는 방법
> 반지름 r인 원에서, 호의 길이 l = 반지름 r 일 때, 그 중심각을 1 라디안 (rad) 이라고 한다.

cv2.getRotationMatrix2D() 함수를 사용한다.

```
mtrx = cv2.getRotationMatrix2D(center, angle, scale)
```
`center` : 회전축 중심 좌표 (x, y)
`angle` : 회전할 각도, 60진법
`scale` : 확대 및 축소비율

</div>
</details>

## 2. 이미지 뒤틀기

1. **어핀 변환(Affine Transform)**

**어핀 변환이란?**

뒤틀기 방법 중 하나로 이미지에 좌표를 지정한 후 그 좌표 값을 원하는 좌표로 이동하며 이미지를 뒤트는 방법 (2차원)

```
martix = cv2.getAffineTransform(pts1, pts2)

pts1: 변환 전 영상의 좌표 3개, 3 x 2 배열
pts2: 변환 후 영상의 좌표 3개, 3 x 2 배열
matrix: 변환 행렬 반환, 2 x 3 행렬
```

</div>
</details>

2. **원근 변환(Perspective Transform)**

**원근 변환이란?**

원근법의 원리를 적용해 변환하는 방법 (3차원)

```
mtrx = cv2.getPerspectiveTransform(pts1, pts2)

pts1: 변환 이전 영상의 좌표 4개, 4 x 2 배열
pts2: 변환 이후 영상의 좌표 4개, 4 x 2 배열
mtrx: 변환행렬 반환, 3 x 3 행렬
```

## 3. 개인 프로젝트

**목표 : 기울어진 자동차 번호판 이미지를 변환하여 규격화한 후 저장한다.
