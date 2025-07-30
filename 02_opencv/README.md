# openCV 색상 표현 방식와 관심 영역(ROI), 스레시홀딩(Thresholding), 히스토그램(Histogram)

## 목차
1. 색상 표현 방식
2. 관심 영역 (ROI)
3. 스레시홀딩 (Thresholding)
4. 히스토그램(Histogram)
5. 개인 프로젝트
   

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

```
cv2.COLOR_BGR2GRAY: BGR 색상 이미지를 회색조 이미지로 변환

cv2.COLOR_GRAY2BGR: 회색조 이미지를 BGR 색상 이미지로 변환

cv2.COLOR_BGR2RGB: BGR 색상 이미지를 RGB 색상 이미지로 변환

cv2.COLOR_BGR2HSV: BGR 색상 이미지를 HSV 색상 이미지로 변환

cv2.COLOR_HSV2BGR: HSV 색상 이미지를 BGR 색상 이미지로 변환

cv2.COLOR_BGR2YUV: BGR 색상 이미지를 YUV 색상 이미지로 변환

cv2.COLOR_YUV2BGR: YUB 색상 이미지를 BGR 색상 이미지로 변환
```


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
<img width="300" height="99" alt="image" src="https://github.com/user-attachments/assets/5b82532b-7040-407f-b82c-e50b06965336" />

빨강을 BGR로 표현하면 (0, 0, 255)이지만 HSV로 표현하면 (0, 255, 255)로 표현된다.

RGB나 BGR 방식은 세가지 채널의 값을 모두 알아야 색상을 알 수 있지만, HSV 방식은 H(색조)값만 알면 바로 알 수 있다.


## 2.관심 영역 (ROI)

말 그대로 이미지 내에서 관심 있는 영역을 말한다.

<예제 이미지>

<img width="600" height="338" alt="image" src="https://github.com/user-attachments/assets/72ebd297-0795-46d6-90de-51f66b453634" />


1. **관심 영역 표시**
```python3
import cv2
import numpy as np

img = cv2.imread('./img/sunset.jpg')

x=320; y=150; w=50; h=50        # roi 좌표
roi = img[y:y+h, x:x+w]         # roi 지정        ---①

print(roi.shape)                # roi shape, (50,50,3)
cv2.rectangle(roi, (0,0), (h-1, w-1), (0,255,0)) # roi 전체에 사각형 그리기 ---②
cv2.imshow("img", img)

key = cv2.waitKey(0)
print(key)
cv2.destroyAllWindows()
```
<img width="896" height="548" alt="image" src="https://github.com/user-attachments/assets/65fb800a-6adb-4fc4-af7a-219ce2976212" />


2. **관심 영역 복제 및 새 창에 띄우기**
```python3
import cv2
import numpy as np

img = cv2.imread('../img/sunset.jpg')

x=320; y=150; w=50; h=50
roi = img[y:y+h, x:x+w]     # roi 지정
img2 = roi.copy()           # roi 배열 복제 ---①

img[y:y+h, x+w:x+w+w] = roi # 새로운 좌표에 roi 추가, 태양 2개 만들기
cv2.rectangle(img, (x,y), (x+w+w, y+h), (0,255,0)) # 2개의 태양 영역에 사각형 표시

cv2.imshow("img", img)      # 원본 이미지 출력
cv2.imshow("roi", img2)     # roi 만 따로 출력

cv2.waitKey(0)
cv2.destroyAllWindows()
```
<img width="900" height="559" alt="image" src="https://github.com/user-attachments/assets/9e9c1c8f-2a0d-4758-a32b-840ee429aa36" />


3. **마우스 이벤트로 관심 영역 지정, 표시, 저장**
```python3
import cv2
import numpy as np

isDragging = False                      # 마우스 드래그 상태 저장 
x0, y0, w, h = -1,-1,-1,-1              # 영역 선택 좌표 저장
blue, red = (255,0,0),(0,0,255)         # 색상 값 

def onMouse(event,x,y,flags,param):     # 마우스 이벤트 핸들 함수  ---①
    global isDragging, x0, y0, img      # 전역변수 참조
    if event == cv2.EVENT_LBUTTONDOWN:  # 왼쪽 마우스 버튼 다운, 드래그 시작 ---②
        isDragging = True
        x0 = x
        y0 = y
    elif event == cv2.EVENT_MOUSEMOVE:  # 마우스 움직임 ---③
        if isDragging:                  # 드래그 진행 중
            img_draw = img.copy()       # 사각형 그림 표현을 위한 이미지 복제
            cv2.rectangle(img_draw, (x0, y0), (x, y), blue, 2) # 드래그 진행 영역 표시
            cv2.imshow('img', img_draw) # 사각형 표시된 그림 화면 출력
    elif event == cv2.EVENT_LBUTTONUP:  # 왼쪽 마우스 버튼 업 ---④
        if isDragging:                  # 드래그 중지
            isDragging = False          
            w = x - x0                  # 드래그 영역 폭 계산
            h = y - y0                  # 드래그 영역 높이 계산
            print("x:%d, y:%d, w:%d, h:%d" % (x0, y0, w, h))
            if w > 0 and h > 0:         # 폭과 높이가 양수이면 드래그 방향이 옳음 ---⑤
                img_draw = img.copy()   # 선택 영역에 사각형 그림을 표시할 이미지 복제
                # 선택 영역에 빨간 사각형 표시
                cv2.rectangle(img_draw, (x0, y0), (x, y), red, 2) 
                cv2.imshow('img', img_draw) # 빨간 사각형 그려진 이미지 화면 출력
                roi = img[y0:y0+h, x0:x0+w] # 원본 이미지에서 선택 영영만 ROI로 지정 ---⑥
                cv2.imshow('cropped', roi)  # ROI 지정 영역을 새창으로 표시
                cv2.moveWindow('cropped', 0, 0) # 새창을 화면 좌측 상단에 이동
                cv2.imwrite('./cropped.jpg', roi)   # ROI 영역만 파일로 저장 ---⑦
                print("croped.")
            else:
                cv2.imshow('img', img)  # 드래그 방향이 잘못된 경우 사각형 그림ㅇㅣ 없는 원본 이미지 출력
                print("좌측 상단에서 우측 하단으로 영역을 드래그 하세요.")

img = cv2.imread('../img/sunset.jpg')
cv2.imshow('img', img)
cv2.setMouseCallback('img', onMouse) # 마우스 이벤트 등록 ---⑧
cv2.waitKey()
cv2.destroyAllWindows()
```
<img width="900" height="547" alt="image" src="https://github.com/user-attachments/assets/6dca89ff-2ae8-4b2f-a3fc-f5dedef52e96" />


## 3. 
