# 컨투어 (Contour)

## 목차

1. 컨투어 (Contour)
   - 컨투어란?
   - 컨투어 찾기와 그리기
   - 트리 계층의 컨투어 그리기
   - 컨투어 단순화
   - 컨투어 볼록 선체 (Convex hull)

## 1. 컨투어(Contour)
<details>
<summary></summary>
<div markdown="1">

## **1-1.컨투어란?**

지형의 높이가 같은 영역을 하나로 표시한 **등고선**을 의미한다.

영상에서 컨투어를 그리면 인식된 객체의 형태를 쉽게 인식 할 수 있다.

contours, hierarchy = cv2.findContours() 함수를 사용한다.
```
contours, hierarchy = cv2.findContours(src, mode, method, contours, hierarchy, offset)
```
`src` : 입력 영상, 검정과 흰색으로 구성된 바이너리 이미지

`mode` : 컨투어 제공 방식 (cv2.RETR_EXTERNAL: 가장 바깥쪽 라인만 생성, cv2.RETR_LIST: 모든 라인을 계층 없이 생성, cv2.RET_CCOMP: 모든 라인을 2 계층으로 생성, cv2.RETR_TREE: 모든 라인의 모든 계층 정보를 트리 구조로 생성)

`method` : 근사 값 방식 (cv2.CHAIN_APPROX_NONE: 근사 없이 모든 좌표 제공, cv2.CHAIN_APPROX_SIMPLE: 컨투어 꼭짓점 좌표만 제공, cv2.CHAIN_APPROX_TC89_L1: Teh-Chin 알고리즘으로 좌표 개수 축소, cv2.CHAIN_APPROX_TC89_KCOS: Teh-Chin 알고리즘으로 좌표 개수 축소)

`contours(optional)` : 검출한 컨투어 좌표 (list type)

`hierarchy(optional)` : 컨투어 계층 정보 (Next, Prev, FirstChild, Parent, -1 [해당 없음])

`offset(optional)` : ROI 등으로 인해 이동한 컨투어 좌표의 오프셋

<br><br>

컨투어를 그리기 위해서는 cv2.drawContours() 함수를 사용한다.
```
cv2.drawContours(img, contours, contourIdx, color, thickness)
```
`img` : 입력 영상

`contours` : 그림 그릴 컨투어 배열 (cv2.findContours() 함수의 반환 결과를 전달해주면 됨)

`contourIdx` : 그림 그릴 컨투어 인덱스, -1: 모든 컨투어 표시

`color` : 색상 값

`thickness` : 선 두께, 0: 채우기

<br><br>

## **1-2. 컨투어 찾기와 그리기**

```python3
# 컨투어 찾기와 그리기 (cntr_find.py)

import cv2
import numpy as np

img = cv2.imread('../img/shapes.png')
img2 = img.copy()

# @그레이스케일로 변환
imgray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# @스레시홀드로 바이너리 이미지로 만들어서 검은배경에 흰색전경으로 반전
ret, imthres = cv2.threshold(imgray, 127, 255, cv2.THRESH_BINARY_INV)

# @가장 바깥쪽 컨투어에 대해 모든 좌표 반환 / openCV 4.x 이상 버전에는 img값을 리턴하지 않음
contour, hierarchy = cv2.findContours(imthres, cv2.RETR_EXTERNAL, \
                                                 cv2.CHAIN_APPROX_NONE)  # 모든 좌표에 컨투어 표시

# @가장 바깥쪽 컨투어에 대해 꼭지점 좌표만 반환 / openCV 4.x 이상 버전에는 img값을 리턴하지 않음
contour2, hierarchy = cv2.findContours(imthres, cv2.RETR_EXTERNAL, \
                                                cv2.CHAIN_APPROX_SIMPLE)  #꼭짓점에만 컨투어 표시
# @각각의 컨투의 갯수 출력
print('도형의 갯수: %d(%d)'% (len(contour), len(contour2)))

# @모든 좌표를 갖는 컨투어 그리기, 초록색
cv2.drawContours(img, contour, -1, (0,255,0), 4)
# @꼭지점 좌표만을 갖는 컨투어 그리기, 초록색
cv2.drawContours(img2, contour2, -1, (0,255,0), 4)

# @컨투어 모든 좌표를 작은 파랑색 점(원)으로 표시
for i in contour:
    for j in i:
        cv2.circle(img, tuple(j[0]), 1, (255,0,0), -1) 

# @컨투어 꼭지점 좌표를 작은 파랑색 점(원)으로 표시
for i in contour2:
    for j in i:
        cv2.circle(img2, tuple(j[0]), 1, (255,0,0), -1) 

# @결과 출력
cv2.imshow('CHAIN_APPROX_NONE', img)
cv2.imshow('CHAIN_APPROX_SIMPLE', img2)

cv2.waitKey(0)
cv2.destroyAllWindows()
```
<img width="1273" height="270" alt="image" src="https://github.com/user-attachments/assets/6ea67ce4-f030-43b7-8692-384ac519ce4a" />



## **1-3. 트리 계층의 컨투어 그리기**

**트리계층 컨투어 : 이미지 속에 여러 윤곽선이 있을 경우, 포함 관계에 따라 트리 구조로 컨투어를 만드는 것**

```python3
# 컨투어 계층 트리

import cv2
import numpy as np

# @영상 읽기
img = cv2.imread('../img/shapes_donut.png')
img2 = img.copy()
# @바이너리 이미지로 변환
imgray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
ret, imthres = cv2.threshold(imgray, 127, 255, cv2.THRESH_BINARY_INV)

# @가장 바깥 컨투어만 수집
contour, hierarchy = cv2.findContours(imthres, cv2.RETR_EXTERNAL, \
                                                cv2.CHAIN_APPROX_NONE)
# @가장 바깥 컨투어 갯수와 계층 트리 출력
print(len(contour), hierarchy)
'''
3 [[[ 1 -1 -1 -1]
  [ 2  0 -1 -1]
  [-1  1 -1 -1]]]
'''

# @모든 컨투어를 트리 계층 으로 수집
contour2, hierarchy = cv2.findContours(imthres, cv2.RETR_TREE, \
                                            cv2.CHAIN_APPROX_SIMPLE)
# @모든 컨투어 갯수와 계층 트리 출력
print(len(contour2), hierarchy)
'''
6 [[[ 2 -1  1 -1]
  [-1 -1 -1  0]
  [ 4  0  3 -1]
  [-1 -1 -1  2]
  [-1  2  5 -1]
  [-1 -1 -1  4]]]
'''

# @가장 바깥 컨투어만 그리기
cv2.drawContours(img, contour, -1, (0,255,0), 3)
# @모든 컨투어 그리기
for idx, cont in enumerate(contour2): 
    # 랜덤한 컬러 추출
    color = [int(i) for i in np.random.randint(0,255, 3)]
    # 컨투어 인덱스 마다 랜덤한 색상으로 그리기
    cv2.drawContours(img2, contour2, idx, color, 3)
    # 컨투어 첫 좌표에 인덱스 숫자 표시
    cv2.putText(img2, str(idx), tuple(cont[0][0]), cv2.FONT_HERSHEY_PLAIN, \
                                                            1, (0,0,255))

# @결과 출력
cv2.imshow('RETR_EXTERNAL', img)
cv2.imshow('RETR_TREE', img2)

cv2.waitKey(0)
cv2.destroyAllWindows()
```
<img width="1271" height="279" alt="image" src="https://github.com/user-attachments/assets/f1ee601d-25d8-4580-bf04-4cdf360e7d5c" />



## **1-4. 컨투어 단순화**

**이미지가 가지고 있는 경계의 노이즈를 단순화한 이미지를 컨투어를 만드는 것**

cv2.approxPolyDP() 함수를 사용한다.
```
approx = cv2.approxPolyDP(contour, epsilon, closed)
```
`contour` : 대상 컨투어 좌표

`epsilon` : 근사 값 정확도, 오차 범위

`closed` : 컨투어의 닫힘 여부

`approx` : 근사 계산한 컨투어 좌표

<br><br>

```python3
# 근사 컨투어

import cv2
import numpy as np

img = cv2.imread('../img/bad_rect.png')
img2 = img.copy()

# @그레이스케일과 바이너리 스케일 변환
imgray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) 
ret, th = cv2.threshold(imgray, 127, 255, cv2.THRESH_BINARY)

# @컨투어 찾기
contours, hierachy = cv2.findContours(th, cv2.RETR_EXTERNAL, \
                                     cv2.CHAIN_APPROX_SIMPLE)
contour = contours[0]

# @전체 둘레의 0.05로 오차 범위 지정
epsilon = 0.05 * cv2.arcLength(contour, True)
# @근사 컨투어 계산
approx = cv2.approxPolyDP(contour, epsilon, True)

# @각각 컨투어 선 그리기
cv2.drawContours(img, [contour], -1, (0,255,0), 3)
cv2.drawContours(img2, [approx], -1, (0,255,0), 3)

# @결과 출력
cv2.imshow('contour', img)
cv2.imshow('approx', img2)

cv2.waitKey()
cv2.destroyAllWindows()
```
<img width="1015" height="330" alt="image" src="https://github.com/user-attachments/assets/6b5c7358-bfcd-41ce-b65b-c28aa757211e" />



## **1-6. 컨투어 볼록 선체 (Convex hull)**

대상을 **완전하게 포함하는 외곽 영역**을 찾는 방법

cv2.convexHull() 볼록 선체 게산 함수
```
hull = cv2.convexHull(points, hull, clockwise, returnPoints)
```
`points` : 입력 컨투어

`hull(optional` ): 볼록 선체 결과

`clockwise(optional)` : 방향 지정 (True: 시계 방향)

`returnPoints(optional)` : 결과 좌표 형식 선택 (True: 볼록 선체 좌표 변환, False: 입력 컨투어 중에 볼록 선체에 해당하는 인덱스 반환)

<br><br>

cv2.isContourConvex() : 볼록 선체 만족 여부 확인 함수
```
retval = cv2.isContourConvex(contour)
```
`retval` : True인 경우 볼록 선체임

<br><br>

cv2.convexityDefects() : 볼록 선체 결함 찾는 함수
```
defects = cv2.convexityDefects(contour, convexhull)
```
`contour` : 입력 컨투어

`convexhull` : 볼록 선체에 해당하는 컨투어의 인덱스

`defects` : 볼록 선체 결함이 있는 컨투어의 배열 인덱스, N x 1 x 4 배열, [starts, end, farthest, distance]

`start` : 오목한 각이 시작되는 컨투어의 인덱스

`end` : 오목한 각이 끝나는 컨투어의 인덱스

`farthest` : 볼록 선체에서 가장 먼 오목한 지점의 컨투어 인덱스

`distance` : farthest와 볼록 선체와의 거리

<br><br>

```python3
# 볼록 선체

import cv2
import numpy as np

img = cv2.imread('../img/hand.jpg')
img2 = img.copy()

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
ret, th = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)

# @컨투어 찾기와 그리기
contours, heiarchy = cv2.findContours(th, cv2.RETR_EXTERNAL, \
                                         cv2.CHAIN_APPROX_SIMPLE)
cntr = contours[0]
cv2.drawContours(img, [cntr], -1, (0, 255,0), 1)

# @볼록 선체 찾기(좌표 기준)와 그리기
hull = cv2.convexHull(cntr)
cv2.drawContours(img2, [hull], -1, (0,255,0), 1)

# @볼록 선체 만족 여부 확인
print(cv2.isContourConvex(cntr), cv2.isContourConvex(hull))

# @볼록 선체 찾기(인덱스 기준)
hull2 = cv2.convexHull(cntr, returnPoints=False)

# @볼록 선체 결함 찾기
defects = cv2.convexityDefects(cntr, hull2)
# 볼록 선체 결함 순회
for i in range(defects.shape[0]):
    # 시작, 종료, 가장 먼 지점, 거리
    startP, endP, farthestP, distance = defects[i, 0]
    # 가장 먼 지점의 좌표 구하기
    farthest = tuple(cntr[farthestP][0])
    # 거리를 부동 소수점으로 변환
    dist = distance/256.0
    # 거리가 1보다 큰 경우
    if dist > 1 :
        # 빨강색 점 표시 
        cv2.circle(img2, farthest, 3, (0,0,255), -1)

# @결과 출력
cv2.imshow('contour', img)
cv2.imshow('convex hull', img2)

cv2.waitKey(0)
cv2.destroyAllWindows()
```
<img width="809" height="481" alt="image" src="https://github.com/user-attachments/assets/6b59a789-9ae6-4f2e-bb54-2c8f4f55da0c" />
