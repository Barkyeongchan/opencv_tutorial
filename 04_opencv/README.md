# 이미지 제어와 이미지 뒤틀기 / 영상 필터와 블러링 / 경계 검출

## 목차
1. 이미지 제어
2. 이미지 뒤틀기
3. 개인 프로젝트 (자동차 번호판 추출)
4. 영상 필터와 블러링
5. 경계 검출
6. 개인 프로젝트 (번호판 경계 검출)

## 1. 이미지 제어
<details>
<summary></summary>
<div markdown="1">

## 1-1. **이미지 이동(Translation)**

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



## 1-2. **이미지 확대/축소(Scaling)**

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



## 1-3. **이미지 회전(Rotation)**

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
<details>
<summary></summary>
<div markdown="1">
  
## 2-1. **어핀 변환(Affine Transform)**

**어핀 변환이란?**

뒤틀기 방법 중 하나로 이미지에 좌표를 지정한 후 그 좌표 값을 원하는 좌표로 이동하며 이미지를 뒤트는 방법 (2차원)

 cv2.getAffineTransform() 함수를 사용한다.

```
martix = cv2.getAffineTransform(pts1, pts2)
```

`pts1` : 변환 전 영상의 좌표 3개, 3 x 2 배열

`pts2` : 변환 후 영상의 좌표 3개, 3 x 2 배열

`matrix` : 변환 행렬 반환, 2 x 3 행렬

```python3
# 어핀(Affine) 변환

import cv2
import numpy as np
from matplotlib import pyplot as plt

file_name = '../img/fish.jpg'
img = cv2.imread(file_name)
rows, cols = img.shape[:2]  # 영상 크기 제어

# @변환 전, 후 각 3개의 좌표 생성
pts1 = np.float32([[100, 50], [200, 50], [100, 200]])
pts2 = np.float32([[80, 70], [210, 60], [250, 120]])

# @변환 전 좌표를 이미지에 표시
cv2.circle(img, (100, 50), 5, (255, 0, 0), -1)
cv2.circle(img, (200, 50), 5, (0, 255, 0), -1)
cv2.circle(img, (100, 200), 5, (0, 0, 255), -1)

# @짝지은 3개의 좌표로 변환 행렬 계산
mtrx = cv2.getAffineTransform(pts1, pts2)

# #어핀 변환 적용
dst = cv2.warpAffine(img, mtrx, (int(cols*1.5), rows))

# @이미지 출력
cv2.imshow('origin',img)
cv2.imshow('affin', dst)

cv2.waitKey(0)
cv2.destroyAllWindows()
```

<img width="959" height="426" alt="image" src="https://github.com/user-attachments/assets/b920ac99-035f-4a46-a927-ac27609389d3" />



## 2-2. **원근 변환(Perspective Transform)**

**원근 변환이란?**

원근법의 원리를 적용해 변환하는 방법 (3차원)

cv2.getPerspectiveTransform() 함수를 사용한다.

```
mtrx = cv2.getPerspectiveTransform(pts1, pts2)
```

`pts1` : 변환 이전 영상의 좌표 4개, 4 x 2 배열

`pts2` : 변환 이후 영상의 좌표 4개, 4 x 2 배열

`mtrx` : 변환행렬 반환, 3 x 3 행렬

```python3
# 원근(Perspective) 변환

import cv2
import numpy as np

file_name = "../img/fish.jpg"
img = cv2.imread(file_name)
rows, cols = img.shape[:2]

# @원근 변환 전 후 4개 좌표
pts1 = np.float32([[0,0], [0,rows], [cols, 0], [cols,rows]])
pts2 = np.float32([[100,50], [10,rows-50], [cols-100, 50], [cols-10,rows-50]])

# @변환 전 좌표를 원본 이미지에 표시
cv2.circle(img, (0,0), 10, (255,0,0), -1)
cv2.circle(img, (0,rows), 10, (0,255,0), -1)
cv2.circle(img, (cols,0), 10, (0,0,255), -1)
cv2.circle(img, (cols,rows), 10, (0,255,255), -1)

# @원근 변환 행렬 계산
mtrx = cv2.getPerspectiveTransform(pts1, pts2)

# @원근 변환 적용
dst = cv2.warpPerspective(img, mtrx, (cols, rows))

cv2.imshow("origin", img)
cv2.imshow('perspective', dst)
cv2.waitKey(0)
cv2.destroyAllWindows()
```

<img width="770" height="428" alt="image" src="https://github.com/user-attachments/assets/56e5960b-8604-414c-80fc-f8eb2a27c7cd" />



## 2-3. **마우스와 원근 변환을 사용해 문서 스캔 효과 만들기**

```python3
# 마우스 이벤트로 원근 변환을 사용해 문서 스캔효과 내기

import cv2
import numpy as np

# @변수 정의
win_name = "scanning"
img = cv2.imread("../img/paper.jpg")
rows, cols = img.shape[:2]
draw = img.copy()
pts_cnt = 0
pts = np.zeros((4,2), dtype=np.float32)

# @마우스 이벤트 함수
def onMouse(event, x, y, flags, param):  # 마우스 이벤트 콜백 함수 구현
    global  pts_cnt                      # 마우스로 찍은 좌표의 갯수 저장
    if event == cv2.EVENT_LBUTTONDOWN:  
        cv2.circle(draw, (x,y), 10, (0,255,0), -1) # 좌표에 초록색 동그라미 표시
        cv2.imshow(win_name, draw)

        pts[pts_cnt] = [x,y]            # 마우스 좌표 저장
        pts_cnt+=1
        
        if pts_cnt == 4:                       # 좌표가 4개 수집됨 
            # 좌표 4개 중 상하좌우 찾기 ---② 
            sm = pts.sum(axis=1)                 # 4쌍의 좌표 각각 x+y 계산
            diff = np.diff(pts, axis = 1)        # 4쌍의 좌표 각각 x-y 계산

            topLeft = pts[np.argmin(sm)]         # x+y가 가장 작은 값이 좌상단 좌표
            bottomRight = pts[np.argmax(sm)]     # x+y가 가장 큰 값이 우하단 좌표
            topRight = pts[np.argmin(diff)]      # x-y가 가장 작은 것이 우상단 좌표
            bottomLeft = pts[np.argmax(diff)]    # x-y가 가장 큰 값이 좌하단 좌표

            # 변환 전 4개 좌표 
            pts1 = np.float32([topLeft, topRight, bottomRight , bottomLeft])

            # 변환 후 영상에 사용할 서류의 폭과 높이 계산 
            w1 = abs(bottomRight[0] - bottomLeft[0])    # 상단 좌우 좌표간의 거리
            w2 = abs(topRight[0] - topLeft[0])          # 하당 좌우 좌표간의 거리
            h1 = abs(topRight[1] - bottomRight[1])      # 우측 상하 좌표간의 거리
            h2 = abs(topLeft[1] - bottomLeft[1])        # 좌측 상하 좌표간의 거리
            width = max([w1, w2])                       # 두 좌우 거리간의 최대값이 서류의 폭
            height = max([h1, h2])                      # 두 상하 거리간의 최대값이 서류의 높이
            
            # 변환 후 4개 좌표
            pts2 = np.float32([[0,0], [width-1,0], 
                                [width-1,height-1], [0,height-1]])

            # 변환 행렬 계산 
            mtrx = cv2.getPerspectiveTransform(pts1, pts2)
            # 원근 변환 적용
            result = cv2.warpPerspective(img, mtrx, (int(width), int(height)))
            cv2.imshow('scanned', result)

# @이미지 출력            
cv2.imshow(win_name, img)
cv2.setMouseCallback(win_name, onMouse)    # 마우스 콜백 함수를 GUI 윈도우에 등록

cv2.waitKey(0)
cv2.destroyAllWindows()
```

<img width="1280" height="920" alt="image" src="https://github.com/user-attachments/assets/8fa77abd-32ae-4ef4-9675-d0e6d2c5e6c2" />


</div>
</details>

## 3. 개인 프로젝트 (자동차 번호판 추출)
<details>
<summary></summary>
<div markdown="1">

**목표 : 기울어진 자동차 번호판 이미지를 변환하여 규격화한 후 저장한다.

```python3
# 자동차 번호판 추출

import cv2
import numpy as np
import datetime
import os

max_img = input('추출하려는 이미지 개수를 입력해 주세요.(최대 5): ')
for i in range(1, int(max_img)+1):
    # @변수 정의
    full_path = '../img/car_0' + str(i) +'.jpg'
    car_plate = cv2.imread(full_path)

    if car_plate is None:
        print("❌ car_plate01 이미지가 제대로 불러와지지 않았습니다.")
        exit()

    win_name = "License Plate Extractor"
    rows, cols = car_plate.shape[:2]
    draw = car_plate.copy()
    pts_cnt = 0
    pts = np.zeros((4,2), dtype=np.float32)

    # @마우스 이벤트 함수
    def onMouse(event, x, y, flags, param):  # 마우스 이벤트 콜백 함수 구현
        global  pts_cnt                      # 마우스로 찍은 좌표의 갯수 저장
        if event == cv2.EVENT_LBUTTONDOWN:   # 마우스 왼쪽 버튼 클릭시
            # 1. 클릭 지점에 원 그리기
            cv2.circle(draw, (x,y), 5, (0,255,0), -1)  # 클릭 지점에 녹색 5px의 원 그리기
            cv2.imshow(win_name, draw)

            # 2. 좌표 배열에 마우스 좌표 저장
            pts[pts_cnt] = [x,y]

            # 3. 카운터 증가
            pts_cnt+=1

            # 4. 4개 점 완성시 변환 실행
            if pts_cnt == 4:                       # 좌표가 4개 수집됨

                # @좌표 정렬 알고리즘 이해
                # 좌표 4개 중 상하좌우 찾기
                sm = pts.sum(axis=1)                 # 4쌍의 좌표 각각 x+y 계산
                diff = np.diff(pts, axis = 1)        # 4쌍의 좌표 각각 x-y 계산

                topLeft = pts[np.argmin(sm)]         # x+y가 가장 작은 값이 좌상단 좌표
                bottomRight = pts[np.argmax(sm)]     # x+y가 가장 큰 값이 우하단 좌표
                topRight = pts[np.argmin(diff)]      # x-y가 가장 작은 것이 우상단 좌표
                bottomLeft = pts[np.argmax(diff)]    # x-y가 가장 큰 값이 좌하단 좌표

                # 변환 전 4개 좌표 
                pts1 = np.float32([topLeft, topRight, bottomRight , bottomLeft])

                '''
                한국 번호판 표준 규격
                일반 승용차: 가로 335mm × 세로 170mm (약 2:1 비율)
                대형차량: 가로 440mm × 세로 220mm (2:1 비율)
                픽셀 변환: 300×150 또는 400×200 권장
                '''
                # 번호판 치수 계산 - 한국 표준 규격에 따른 2:1비율 사용
                width = 300
                height = 150

                # 변환 후 4개 좌표
                pts2 = np.float32([[0,0], [width-1,0], 
                                    [width-1,height-1], [0,height-1]])

                # 변환 행렬 계산 
                mtrx = cv2.getPerspectiveTransform(pts1, pts2)
                # 원근 변환 적용
                result = cv2.warpPerspective(car_plate, mtrx, (int(width), int(height)))

                # @파일 저장 기능 구현

                # 1. 저장 경로 처리
                save_dir = "../extracted_plates"   # 저장 폴더가 없으면 생성
                if not os.path.exists(save_dir):
                    os.makedirs(save_dir)

                # 2. 타임 스탬프 기반
                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                filename_time = f"../extracted_plates/plate_{timestamp}.png"    # png형식 선택

                # 3. 순번 기반
                existing_files = len(os.listdir(save_dir))
                filename_os = f"../extracted_plates/plate_{existing_files+1:03d}.png" # png형식 선택

                success = cv2.imwrite(filename_time, result) # 타임 스탬프 파일 저장
                if success:
                    print(f"번호판 저장 완료: {filename_time}")
                    cv2.imshow('Extracted Plate', result)
                else:
                    print("저장 실패!")

                success = cv2.imwrite(filename_os, result) # 순번 파일 저장
                if success:
                    print(f"번호판 저장 완료: {filename_os}")
                    cv2.imshow('Extracted Plate', result)
                else:
                    print("저장 실패!")

    cv2.imshow(win_name, car_plate)
    cv2.setMouseCallback(win_name, onMouse)

    cv2.waitKey(0)
    cv2.destroyAllWindows()
```

</div>
</details>

## 4. 영상 필터와 블러링링
<details>
<summary></summary>
<div markdown="1">

## **4-1. 컨볼루션 연산 (Convolution Operation)**

입력 신호(또는 이미지)에 **커널(또는 필터)**을 슬라이딩하면서 곱하고 더해서 출력하는 연산

<img width="772" height="527" alt="image" src="https://github.com/user-attachments/assets/81a166c0-3044-429a-a8f4-94d0fa48ea83" />



cv2.filter2D() 함수를 사용한다.

```
dst = cv2.filter2D(src, ddepth, kernel, dst, anchor, delta, borderType)
```

`src` : 입력 영상, Numpy 배열

`ddepth` : 출력 영상의 dtype (-1: 입력 영상과 동일)

`kernel` : 컨볼루션 커널, float32의 n x n 크기 배열

`dst(optional)` : 결과 영상

`anchor(optional)` : 커널의 기준점, default: 중심점 (-1, -1)

`delta(optional)` : 필터가 적용된 결과에 추가할 값

`borderType(optional)` : 외곽 픽셀 보정 방법 지정



## **4-2. 평균 블러링 (Average Blurring)**

주변 픽셀 값의 평균을 적용하여 영상을 흐릿하게 하는 작업

cv2.blur()와 cv2.boxFilter() 함수를 사용한다.
```
dst = cv2.blur(src, ksize, dst, anchor, borderType)
```
`src` : 입력 영상, numpy 배열

`ksize` : 커널의 크기

`나머지 파라미터는 cv2.filter2D()와 동일`



```
dst = cv2.boxFilter(src, ddepth, ksize, dst, anchor, normalize, borderType)
```
`ddepth` : 출력 영상의 dtype (-1: 입력 영상과 동일)

`normalize(optional)` : 커널 크기로 정규화(1/ksize²) 지정 여부 (Boolean), default=True

`나머지 파라미터는 cv2.filter2D()와 동일`

 ```python3
# 평균 필터를 생성하여 블러 적용

import cv2
import numpy as np

img = cv2.imread('../img/paper.jpg')
'''
# @5x5 평균 필터 커널 생성
kernel = np.array([[0.04, 0.04, 0.04, 0.04, 0.04],
                   [0.04, 0.04, 0.04, 0.04, 0.04],
                   [0.04, 0.04, 0.04, 0.04, 0.04],
                   [0.04, 0.04, 0.04, 0.04, 0.04],
                   [0.04, 0.04, 0.04, 0.04, 0.04]])
'''
# @5X5 평균 필터 커널 생성
kernel = np.ones((5,5))/5**2

# @필터 적용
blured = cv2.filter2D(img, -1, kernel)

# @이미지 출력
cv2.imshow('origin', img)
cv2.imshow('avrg blur', blured)

# @blur() 함수로 블러링
blur1 = cv2.blur(img, (10,10))
# @boxFilter() 함수로 블러링 적용
blur2 = cv2.boxFilter(img, -1, (10,10))

merged = np.hstack( (img, blur1, blur2))
cv2.imshow('blur', merged)

cv2.waitKey()
cv2.destroyAllWindows()
```

<img width="1280" height="516" alt="image" src="https://github.com/user-attachments/assets/c32d668d-25d6-43b4-ae72-a1ae09690d27" />

<img width="1280" height="505" alt="image" src="https://github.com/user-attachments/assets/744cacc8-c5dc-41df-ab86-eb9bb60f178a" />



## **4-3. 가우시안 블러링(Gaussian Blurring)**

가우시안 분포를 갖는 커널로 블러링 하는 작업

<img width="500" height="213" alt="image" src="https://github.com/user-attachments/assets/15495532-6d31-45ed-82f8-06b04dbfe742" />



cv2.GaussianBlur() 함수와 cv2.getGaussianKernel() 함수를 사용한다.
```
cv2.GaussianBlur(src, ksize, sigmaX, sigmaY, borderType)\
```
`src` : 입력 영상

`ksize` : 커널 크기 (주로 홀수)

`sigmaX` : X 방향 표준편차 (0: auto)

`sigmaY(optional)` : Y 방향 표준편차 (default: sigmaX)

`borderType(optional)` : 외곽 테두리 보정 방식



```
ret = cv2.getGaussianKernel(ksize, sigma, ktype)
```
`ret` : 가우시안 커널 (1차원이므로 ret * ret.T 형태로 사용해야 함)

```python3
# 가우시안 블러링

import cv2
import numpy as np

img = cv2.imread('../img/gaussian_noise.jpg')

# @가우시안 커널을 직접 생성해서 블러링
k1 = np.array([[1, 2, 1],
                   [2, 4, 2],
                   [1, 2, 1]]) *(1/16)
blur1 = cv2.filter2D(img, -1, k1)

# @가우시안 커널을 API로 얻어서 블러링
k2 = cv2.getGaussianKernel(3, 0)
blur2 = cv2.filter2D(img, -1, k2*k2.T)

# @가우시안 블러 API로 블러링 ---③
blur3 = cv2.GaussianBlur(img, (3, 3), 0)

# @이미지 출력
print('k1:', k1)
print('k2:', k2*k2.T)

merged = np.hstack((img, blur1, blur2, blur3))
cv2.imshow('gaussian blur', merged)
cv2.waitKey(0)
cv2.destroyAllWindows()
```

<img width="1280" height="349" alt="image" src="https://github.com/user-attachments/assets/5a312d48-b433-4fee-8bda-cbb52a5003fb" />



## **4-4. 미디언 블러링(Median Blurring)**

커널의 픽셀 값 중 중앙값을 선택하는 것

cv2.medianBlur() 함수를 사용한다.
```
dst = cv2.medianBlur(src, ksize)
```
`src` : 입력 영상

`ksize` : 커널 크기

```python3
# 미디언 블러링

import cv2
import numpy as np

img = cv2.imread("../img/salt_pepper_noise.jpg")

# @미디언 블러 적용
blur = cv2.medianBlur(img, 5)

# @이미지 출력 
merged = np.hstack((img,blur))
cv2.imshow('media', merged)
cv2.waitKey(0)
cv2.destroyAllWindows()
```

<img width="649" height="450" alt="image" src="https://github.com/user-attachments/assets/ea316fec-1b45-46b3-99aa-e9f328101bd6" />



## **4-5. 바이레터럴 필터 (Bilateral Filter)**

가우시안 필터와 경계 필터를 결합하여 노이즈는 줄이고 경계는 선명하게 만들지만 속도가 다소 느린 단점이 있다.

cv2.bilateralFilter() 함수를 사용한다.
```
dst = cv2.bilateralFilter(src, d, sigmaColor, sigmaSpace, dst, borderType)
```
`src` : 입력 영상

`d` : 필터의 직경(diameter), 5보다 크면 매우 느림

`sigmaColor` : 색공간의 시그마 값

`sigmaSpace` : 좌표 공간의 시그마 값

```python3
# 바이레터럴 필터

import cv2
import numpy as np

img = cv2.imread("../img/gaussian_noise.jpg")

# @가우시안 필터 적용
blur1 = cv2.GaussianBlur(img, (5,5), 0)

# @바이레터럴 필터 적용
blur2 = cv2.bilateralFilter(img, 5, 75, 75)

# @이미지 출력
merged = np.hstack((img, blur1, blur2))
cv2.imshow('bilateral', merged)
cv2.waitKey(0)
cv2.destroyAllWindows()
```

<img width="1148" height="425" alt="image" src="https://github.com/user-attachments/assets/90011719-8fd9-47ac-841c-8fe3b5ccabaa" />

</div>
</details>

## 5. 경계 검출
<details>
<summary></summary>
<div markdown="1">

## 5-1. **캐니 엣지 (Canny Edge)**

**다음의 4단계 알고리즘에 따라 경계를 검출한다.**

1. 노이즈 제거: 5 x 5 가우시안 블러링 필터로 노이즈 제거
   
2. 경계 그레디언트 방향 계산: 소벨 필터로 경계 및 그레디언트 방향 검출
   
3. 비최대치 억제(Non-Maximum Suppression): 그레디언트 방향에서 검출된 경계 중 가장 큰 값만 선택하고 나머지는 제거
   
4. 이력 스레시홀딩: 두 개의 경계 값(Max, Min)을 지정해서 경계 영역에 있는 픽셀들 중 큰 경계 값(Max) 밖의 픽셀과 연결성이 없는 픽셀 제거

cv2.Canny() 함수를 사용한다.
```
edges = cv2.Canny(img, threshold1, threshold2, edges, apertureSize, L2gardient)
```
`img` : 입력 영상

`threshold1, threshold2` : 이력 스레시홀딩에 사용할 Min, Max 값

`apertureSize` : 소벨 마스크에 사용할 커널 크기

`L2gradient` : 그레디언트 강도를 구할 방식 (True: 제곱 합의 루트 False: 절댓값의 합)

`edges` : 엣지 결과 값을 갖는 2차원 배열

```python3
# 캐니 엣지

import cv2, time
import numpy as np

img = cv2.imread("../img/sudoku.jpg")

# @케니 엣지 적용 
edges = cv2.Canny(img,100,200)

# @이미지 출력
cv2.imshow('Original', img)
cv2.imshow('Canny', edges)
cv2.waitKey(0)
cv2.destroyAllWindows()
```

<img width="676" height="484" alt="image" src="https://github.com/user-attachments/assets/c66e0b67-274c-47ee-8caa-eab9c9075ae4" />



## **5-2. 모폴로지 (Morphology)**

**모폴로지란?**

'형태학'이라는 뜻으로, 노이즈 제거 / 구멍 채우기 / 끊어진 선 이어 붙이기 등에 쓰이는 형태학적 연산을 말한다.

바이너리 이미지(흑백으로만 이루어진 이미지)에만 적용 할 수 있다.

## **5-3. 침식 연산 (Erosion)**

**이미지를 깍아내는 연산**

0과 1로 이루어진 구조화 요소 커널을 사용한다.

<img width="720" height="300" alt="image" src="https://github.com/user-attachments/assets/ecd874e0-1669-4a37-b8fc-d26a11c4a697" />



구조화 생성을 위해 cv2.getStructuringElement() 함수를 사용한다.
```
cv2.getStructuringElement(shape, ksize, anchor)
```
`shape` : 구조화 요소 커널 모양 (cv2.MORPH_RECT: 사각형, cv2.MORPH_EPLIPSE: 타원형, cv2.MORPH_CROSS: 십자형)

`ksize` : 커널 크기

`anchor(optional)` : 구조화 요소의 기준점, cv2.MORPH_CROSS에만 의미 있으며 기본 값은 중심점 (-1, -1)

위의 구조화 요소 커널로 침식 연산을 수행할 때에는 cv2.erode() 함수를 사용한다.
```
dst = cv2.erode(src, kernel, anchor, iterations, borderType, borderValue)
```
`src` : 입력 영상, 바이너리

`kernel` : 구조화 요소 커널

`anchor(optional)` : cv2.getStructuringElement()와 동일

`iterations(optional)` : 침식 연산 적용 반복 횟수

`boderType(optional)` : 외곽 영역 보정 방법 

`boderValue(optional)` : 외곽 영역 보정 값

```python3
# 침식 연산

import cv2
import numpy as np

img = cv2.imread('../img/morph_dot.png')

# @구조화 요소 커널, 사각형 (3x3) 생성
k = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))

# @침식 연산 적용
erosion = cv2.erode(img, k)

# @이미지 출력
merged = np.hstack((img, erosion))
cv2.imshow('Erode', merged)
cv2.waitKey(0)
cv2.destroyAllWindows()
```

<img width="337" height="273" alt="image" src="https://github.com/user-attachments/assets/fb879369-db5d-4d91-842e-1d51c1e2e43b" />



## **5-4. 팽창 연산 (Dilatation)**

**물체 주변을 확장하는 연산**

<img width="720" height="350" alt="image" src="https://github.com/user-attachments/assets/d3595fa2-1f34-4942-b446-d8f7f9f9fbec" />



cv2.dilate() 함수를 사용한다.
```
dst = cv2.dilate(src, kernel, dst, anchor, iterations, bordeType, borderValue)
```
`모든 파라미터는 cv2.erode()와 동일합니다.`

```python3
# 팽창 연산

import cv2
import numpy as np

img = cv2.imread('../img/morph_hole.png')

# @구조화 요소 커널, 사각형 (3x3) 생성
k = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))

# @팽창 연산 적용 
dst = cv2.dilate(img, k)

# @이미지 출력
merged = np.hstack((img, dst))
cv2.imshow('Dilation', merged)
cv2.waitKey(0)
cv2.destroyAllWindows()
```

<img width="334" height="264" alt="image" src="https://github.com/user-attachments/assets/82964102-b4eb-42cd-8046-150f7b1ef790" />



## **5-5. 열림(Opening)/닫힘(Closing) 연산과 그레디언트 연산 (Gradient)

**열림 연산 : 침식 연산 후 팽창 연산을 적용 = 주변보다 밝은 노이즈를 제거, 맞닿은 독립개체를 분리하거나 돌출된 모양을 제거하는데 효과적.**

**닫힘 연산 : 팽창 연산 후 침식 연산을 적용 = 주변보다 어두운 노이즈를 제거, 끊어진 걔체를 연결하거나 구멍을 메우는데 효과적**

**그레디언트 연산 : 팽창 연산 적용 이미지 - 침식 연산 적용 이미지  = 경계 픽셀만 검출, 경계 검출과 비슷한 결과물을 얻을 수 있다.**

cv2.morphologyEx() 함수를 사용한다.
```
dst = cv2.morphologyEx(src, op, kernel, dst, anchor, iteration, borderType, borderValue)
```
`src` : 입력 영상

`op` : 모폴로지 연산 종류 (cv2.MORPH_OPEN: 열림 연산, cv2.MORPH_COLSE: 닫힘 연산, cv2.MORPH_GRADIENT: 그레디언트 연산, cv2.MORPH_TOPHAT: 탑햇 연산, cv2.MORPH_BLACKHAT: 블랙햇 연산)

`kernel` : 구조화 요소 커널

`dst(optional)` : 결과 영상

`anchor(optional)` : 커널의 기준점

`iteration(optional)` : 연산 반복 횟수

`borderType(optional)` : 외곽 영역 보정 방법

`borderValue(optional)` : 외곽 영역 보정 값

```python3
# 열림과 닫힘 연산으로 노이즈 제거

import cv2
import numpy as np

img1 = cv2.imread('../img/morph_dot.png', cv2.IMREAD_GRAYSCALE)
img2 = cv2.imread('../img/morph_hole.png', cv2.IMREAD_GRAYSCALE)    

# @구조화 요소 커널, 사각형 (5x5) 생성
k = cv2.getStructuringElement(cv2.MORPH_RECT, (5,5))
# @열림 연산 적용
opening = cv2.morphologyEx(img1, cv2.MORPH_OPEN, k)
# @닫힘 연산 적용
closing = cv2.morphologyEx(img2, cv2.MORPH_CLOSE, k)

# @이미지 출력
merged1 = np.hstack((img1, opening))
merged2 = np.hstack((img2, closing))
merged3 = np.vstack((merged1, merged2))
cv2.imshow('opening, closing', merged3)
cv2.waitKey(0)
cv2.destroyAllWindows()
```

<img width="330" height="490" alt="image" src="https://github.com/user-attachments/assets/d2311f66-c3ba-4abf-856c-fbbf1d1ff030" />



```python3
# 모폴로지 그레이언트

import cv2
import numpy as np

img = cv2.imread('../img/morphological.png')

# @구조화 요소 커널, 사각형 (3x3) 생성
k = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
# @열림 연산 적용
gradient = cv2.morphologyEx(img, cv2.MORPH_GRADIENT, k)

# @이미지 출력
merged = np.hstack((img, gradient))
cv2.imshow('gradient', merged)
cv2.waitKey(0)
cv2.destroyAllWindows()
```

<img width="330" height="263" alt="image" src="https://github.com/user-attachments/assets/02fb0eee-b9f4-4a97-9b60-4ce7c12152a3" />

</div>
</details>



