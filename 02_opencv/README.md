<<<<<<< HEAD
# openCV를 활용한 그레이스케일, 쓰레스홀딩
=======
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


## 3. 스레시홀딩 (Thresholding)

바이너리 이미지를 만드는 가장 대표적인 방법

*바이너리 이미지란? 검은색과 흰색으로만 표현된 이미지*

1. **전역 스레시홀딩**

임계값을 임의로 정한 뒤 픽셀값이 임계값을 넘으면 255, 넘지 않으면 0으로 지정하는 방식

```python3
import cv2
import numpy as np
import matplotlib.pylab as plt

img = cv2.imread('../img/gray_gradient.jpg', cv2.IMREAD_GRAYSCALE) #이미지를 그레이 스케일로 읽기

# --- ① NumPy API로 바이너리 이미지 만들기
thresh_np = np.zeros_like(img)   # 원본과 동일한 크기의 0으로 채워진 이미지
thresh_np[ img > 127] = 255      # 127 보다 큰 값만 255로 변경

# ---② OpenCV API로 바이너리 이미지 만들기
ret, thresh_cv = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY) 
print(ret)  # 127.0, 바이너리 이미지에 사용된 문턱 값 반환

# ---③ 원본과 결과물을 matplotlib으로 출력
imgs = {'Original': img, 'NumPy API':thresh_np, 'cv2.threshold': thresh_cv}
for i , (key, value) in enumerate(imgs.items()):
    plt.subplot(1, 3, i+1)
    plt.title(key)
    plt.imshow(value, cmap='gray')
    plt.xticks([]); plt.yticks([])

plt.show()
```
<img width="794" height="248" alt="image" src="https://github.com/user-attachments/assets/054e507f-cdb4-494b-b411-b67354b26f3a" />


2. **스레시홀딩 플래그 (Flag)**
```
cv2.THRESH_BINARY: 픽셀 값이 임계값을 넘으면 value로 지정하고, 넘지 못하면 0으로 지정
cv2.THRESH_BINARY_INV: cv.THRESH_BINARY의 반대
cv2.THRESH_TRUNC: 픽셀 값이 임계값을 넘으면 value로 지정하고, 넘지 못하면 원래 값 유지
cv2.THRESH_TOZERO: 픽셀 값이 임계값을 넘으면 원래 값 유지, 넘지 못하면 0으로 지정
cv2.THRESH_TOZERO_INV: cv2.THRESH_TOZERO의 반대
```

```python3
import cv2
import numpy as np
import matplotlib.pylab as plt

img = cv2.imread('../img/gray_gradient.jpg', cv2.IMREAD_GRAYSCALE)

_, t_bin = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
_, t_bininv = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY_INV)
_, t_truc = cv2.threshold(img, 127, 255, cv2.THRESH_TRUNC)
_, t_2zr = cv2.threshold(img, 127, 255, cv2.THRESH_TOZERO)
_, t_2zrinv = cv2.threshold(img, 127, 255, cv2.THRESH_TOZERO_INV)

imgs = {'origin':img, 'BINARY':t_bin, 'BINARY_INV':t_bininv, \
        'TRUNC':t_truc, 'TOZERO':t_2zr, 'TOZERO_INV':t_2zrinv}
for i, (key, value) in enumerate(imgs.items()):
    plt.subplot(2,3, i+1)
    plt.title(key)
    plt.imshow(value, cmap='gray')
    plt.xticks([]);    plt.yticks([])
    
plt.show()
```
<img width="790" height="511" alt="image" src="https://github.com/user-attachments/assets/4174f2de-6035-45c4-b4a8-c2519b2eea77" />


3. **오츠의 이진화 알고리즘 (Otsu's binarization method)**

임계값을 임의로 정해 두 부류로 나눈 픽셀의 **명암 분포가 가장 균일 할 때의 임계값**을 찾는 알고리즘

```python3
import cv2
import numpy as np
import matplotlib.pylab as plt

# 이미지를 그레이 스케일로 읽기
img = cv2.imread('../img/scaned_paper.jpg', cv2.IMREAD_GRAYSCALE) 
# 경계 값을 130으로 지정  ---①
_, t_130 = cv2.threshold(img, 130, 255, cv2.THRESH_BINARY)        
# 경계 값을 지정하지 않고 OTSU 알고리즘 선택 ---②
t, t_otsu = cv2.threshold(img, -1, 255,  cv2.THRESH_BINARY | cv2.THRESH_OTSU) 
print('otsu threshold:', t)                 # Otsu 알고리즘으로 선택된 경계 값 출력

imgs = {'Original': img, 't:130':t_130, 'otsu:%d'%t: t_otsu}
for i , (key, value) in enumerate(imgs.items()):
    plt.subplot(1, 3, i+1)
    plt.title(key)
    plt.imshow(value, cmap='gray')
    plt.xticks([]); plt.yticks([])

plt.show()
```
<img width="1280" height="488" alt="image" src="https://github.com/user-attachments/assets/d59916da-ca3c-473a-b0fa-c84c4ed25afe" />


4. **적응형 스레시홀딩 (Adaptive)**

이미지를 여러 구역으로 나눈 뒤 나눈 구역 주변의 픽셀값만을 활용하여 임계값을 구하는 방식

조명이 일정하지 않는 이미지도 바이너리 이미지로 만들 수 있다.

```python3
import cv2
import numpy as np 
import matplotlib.pyplot as plt 

blk_size = 9        # 블럭 사이즈
C = 5               # 차감 상수 
img = cv2.imread('../img/sudoku.png', cv2.IMREAD_GRAYSCALE) # 그레이 스케일로  읽기

# ---① 오츠의 알고리즘으로 단일 경계 값을 전체 이미지에 적용
ret, th1 = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

# ---② 어뎁티드 쓰레시홀드를 평균과 가우시안 분포로 각각 적용
th2 = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C,\
                                      cv2.THRESH_BINARY, blk_size, C)
th3 = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, \
                                     cv2.THRESH_BINARY, blk_size, C)

# ---③ 결과를 Matplot으로 출력
imgs = {'Original': img, 'Global-Otsu:%d'%ret:th1, \
        'Adapted-Mean':th2, 'Adapted-Gaussian': th3}
for i, (k, v) in enumerate(imgs.items()):
    plt.subplot(2,2,i+1)
    plt.title(k)
    plt.imshow(v,'gray')
    plt.xticks([]),plt.yticks([])

plt.show()
```
<img width="1048" height="1118" alt="image" src="https://github.com/user-attachments/assets/5122751b-e33a-42da-9ef0-2df8e90cb87e" />


## 4. 히스토그램 (Histogram)

**히스토그램이란?** 

도수 분포표를 그래프로 나타낸 것 즉, 무엇이 몇 개 있는지 개수를 세어놓은 것을 그래프화 한 것

아래와 같은 형식을 갖는다.

```
cv2.calHist(img, channel, mask, histSize, ranges)

img: 이미지 영상, [img]처럼 리스트로 감싸서 전달
channel: 분석 처리할 채널, 리스트로 감싸서 전달 - 1 채널: [0], 2 채널: [0, 1], 3 채널: [0, 1, 2]
mask: 마스크에 지정한 픽셀만 히스토그램 계산, None이면 전체 영역
histSize: 계급(Bin)의 개수, 채널 개수에 맞게 리스트로 표현 - 1 채널: [256], 2 채널: [256, 256], 3 채널: [256, 256, 256]
ranges: 각 픽셀이 가질 수 있는 값의 범위, RGB인 경우 [0, 256]
```

1. **그레이스케일 이미지 히스토그램 (1채널)**

```python3
import cv2
import numpy as np
import matplotlib.pylab as plt

#--① 이미지 그레이 스케일로 읽기 및 출력
img = cv2.imread('../img/mountain.jpg', cv2.IMREAD_GRAYSCALE)
cv2.imshow('img', img)

#--② 히스토그램 계산 및 그리기
hist = cv2.calcHist([img], [0], None, [256], [0,256])
plt.plot(hist)

print("hist.shape:", hist.shape)  #--③ 히스토그램의 shape (256,1)
print("hist.sum():", hist.sum(), "img.shape:",img.shape) #--④ 히스토그램 총 합계와 이미지의 크기
plt.show()
```
<img width="1280" height="505" alt="image" src="https://github.com/user-attachments/assets/d2099e2a-e2d9-48d5-a401-24bfab77a713" />


2. **컬러 이미지를 RGB로 계산한 히스토그램 (3채널)**

```python3
# 색상 이미지 히스토그램 (histo_rgb.py)

import cv2
import numpy as np
import matplotlib.pylab as plt

#--① 이미지 읽기 및 출력
img = cv2.imread('../img/mountain.jpg')
cv2.imshow('img', img)

#--② 히스토그램 계산 및 그리기
channels = cv2.split(img)
colors = ('b', 'g', 'r')
for (ch, color) in zip (channels, colors):
    hist = cv2.calcHist([ch], [0], None, [256], [0, 256])
    plt.plot(hist, color = color)
plt.show()
```
<img width="1280" height="498" alt="image" src="https://github.com/user-attachments/assets/88f34fd6-5bae-479f-8c34-159e14fa6b8f" />


3. **정규화 (Normalization)**

**정규화란?**

특정 영역에 몰려 있는 화지을을 개선하거나, 이미지 간의 연산 조건이 다른 경우 같은 조건으로 만드는 등 이미지를 개선하는 작업

아래와 같은 형식을 갖는다.

```
dst = cv2.normalize(src, dst, alpha, beta, type_flag)

src: 정규화 이전의 데이터
dst: 정규화 이후의 데이터
alpha: 정규화 구간 1
beta: 정규화 구간 2, 구간 정규화가 아닌 경우 사용 안 함
type_flag: 정규화 알고리즘 선택 플래그 상수
```

```python3
import cv2
import numpy as np
import matplotlib.pylab as plt

#--① 그레이 스케일로 영상 읽기
img = cv2.imread('../img/abnormal.jpg', cv2.IMREAD_GRAYSCALE)

#--② 직접 연산한 정규화
img_f = img.astype(np.float32)
img_norm = ((img_f - img_f.min()) * (255) / (img_f.max() - img_f.min()))
img_norm = img_norm.astype(np.uint8)

#--③ OpenCV API를 이용한 정규화
img_norm2 = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)

#--④ 히스토그램 계산
hist = cv2.calcHist([img], [0], None, [256], [0, 255])
hist_norm = cv2.calcHist([img_norm], [0], None, [256], [0, 255])
hist_norm2 = cv2.calcHist([img_norm2], [0], None, [256], [0, 255])

cv2.imshow('Before', img)
cv2.imshow('Manual', img_norm)
cv2.imshow('cv2.normalize()', img_norm2)

hists = {'Before' : hist, 'Manual':hist_norm, 'cv2.normalize()':hist_norm2}
for i, (k, v) in enumerate(hists.items()):
    plt.subplot(1,3,i+1)
    plt.title(k)
    plt.plot(v)
plt.show()
```
<img width="1280" height="613" alt="image" src="https://github.com/user-attachments/assets/da45b145-bab5-4be2-a85b-a6d72090a6ad" />


4. **평탄화 (equalization)**

**평탄화란?**

특정 영역에 집중되어 있는 분포를 전체에 골고루 분포하도록 하는 작업, 명암 대비 개선에 효과적

<img width="300" height="135" alt="image" src="https://github.com/user-attachments/assets/e520ad1d-506b-465b-b35a-f46d062355eb" />

아래와 같은 형식을 갖는다.

```
dst = cv2.equalizeHist(src, dst)

src: 대상 이미지, 8비트 1 채널
dst(optional): 결과 이미지
```

색상 이미지에 적용한 예

```python3
import numpy as np, cv2

img = cv2.imread('../img/yate.jpg') #이미지 읽기, BGR 스케일

#--① 컬러 스케일을 BGR에서 YUV로 변경
img_yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV) 

#--② YUV 컬러 스케일의 첫번째 채널에 대해서 이퀄라이즈 적용
img_yuv[:,:,0] = cv2.equalizeHist(img_yuv[:,:,0]) 

#--③ 컬러 스케일을 YUV에서 BGR로 변경
img2 = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR) 

cv2.imshow('Before', img)
cv2.imshow('After', img2)
cv2.waitKey()
cv2.destroyAllWindows()
```
<img width="1280" height="508" alt="image" src="https://github.com/user-attachments/assets/556ade99-466b-4146-8437-5db81967bc6a" />


5. **CLAHE (Contrast Limited Adaptive Histogram Equalization)**

**CLAHE란?**

<img width="1039" height="513" alt="image" src="https://github.com/user-attachments/assets/9b520a71-99d7-4583-8b46-54ace9e72a28" />

위 이미지처럼 평탄화를 적용한 후 하이라이트 부분의 데이터가 없어지는 경우가 생기는데 (과노출), 이런 현상을 막기 위해

일정 영역을 나누어 평탄화를 진행한다(적응형 스레시홀딩과 유사).

하지만 이런 경우 일정 영역 내에서 극단적은 명암이 있는 경우 노이즈가 생기는데, 이 문제를 피하기 위해

***어떤 영역이든 지정된 제한값을 넘으면 그 픽셀을 다른 영역에 균일하게 배분하여 적용***하는 방법

적응형 스레시홀딩과 평탄화를 섞은 듯한 느낌

<img width="300" height="109" alt="image" src="https://github.com/user-attachments/assets/915f2b4d-4133-4b3d-8b3d-c41be6d8997c" />

아래와 같은 형식을 갖는다.

```
clahe = cv2.createCLAHE(clipLimit, tileGridSize)

clipLimit: 대비(Contrast) 제한 경계 값, default=40.0
tileGridSize: 영역 크기, default=8 x 8
clahe: 생성된 CLAHE 객체

clahe.apply(src): CLAHE 적용

src: 입력 이미지
```

```python3
import cv2
import numpy as np
import matplotlib.pylab as plt

#--①이미지 읽어서 YUV 컬러스페이스로 변경
img = cv2.imread('../img/bright.jpg')
img_yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)

#--② 밝기 채널에 대해서 이퀄라이즈 적용
img_eq = img_yuv.copy()
img_eq[:,:,0] = cv2.equalizeHist(img_eq[:,:,0])
img_eq = cv2.cvtColor(img_eq, cv2.COLOR_YUV2BGR)

#--③ 밝기 채널에 대해서 CLAHE 적용
img_clahe = img_yuv.copy()
clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8)) #CLAHE 생성
img_clahe[:,:,0] = clahe.apply(img_clahe[:,:,0])           #CLAHE 적용
img_clahe = cv2.cvtColor(img_clahe, cv2.COLOR_YUV2BGR)

#--④ 결과 출력
cv2.imshow('Before', img)
cv2.imshow('CLAHE', img_clahe)
cv2.imshow('equalizeHist', img_eq)
cv2.waitKey()
cv2.destroyAllWindows()
```
<img width="1280" height="313" alt="image" src="https://github.com/user-attachments/assets/97b8cb56-b6e7-45d4-9b55-748090b84b09" />


## 5. 개인 프로젝트

목표 : 앞서 배운 색상 표현 방식, 스레시홀딩을 사용하여 이미지 결과물을 비교해 본다.

```python3
import cv2
import numpy as np
import matplotlib.pylab as plt

# @이미지 가져오기
img = cv2.imread('../img/sample.jpg')

# @이미지 리사이징
img = cv2.resize(img, dsize=None, fx=0.2, fy=0.2)

# 이미지 크롭하기 위해 현재 사이즈 구함
h, w = img.shape[:2]
side = min(h, w)
center_x, center_y = w // 2, h // 2
x1 = center_x - side // 2
x2 = center_x + side // 2
y1 = center_y - side // 2
y2 = center_y + side // 2

# @크롭된 이미지
crop = img[y1:y2, x1:x2].copy()

# @이미지 꾸미기 (cropped만 사용)
bgra = cv2.cvtColor(crop, cv2.COLOR_BGR2BGRA)  # 알파채널 추가
gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)  # 흑백 변환

blk_size = 9
C = 5
adap = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                             cv2.THRESH_BINARY, blk_size, C)    

ret, thresh_cv_gray = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

ret, thresh_cv = cv2.threshold(crop, 127, 255, cv2.THRESH_BINARY)

# @결과 출력
imgs = {
    'Original': crop,
    'BGRA': bgra,
    'Gray Scale': gray,
    'Threshold' : thresh_cv,
    'Gray Scalr Threshold' : thresh_cv_gray,
    'Adaptive Threshold': adap
}

for i, (key, value) in enumerate(imgs.items()):
    plt.subplot(3, 2, i + 1)
    plt.title(key)
    # 컬러는 RGB로 바꿔서 보여주고, 흑백은 그대로
    if len(value.shape) == 3 and value.shape[2] == 3:
        plt.imshow(cv2.cvtColor(value, cv2.COLOR_BGR2RGB))
    elif len(value.shape) == 3 and value.shape[2] == 4:
        plt.imshow(cv2.cvtColor(value, cv2.COLOR_BGRA2RGBA))
    else:
        plt.imshow(value, 'gray')
    plt.xticks([]), plt.yticks([])

plt.tight_layout()
plt.show()

<img width="640" height="480" alt="Figure_1" src="https://github.com/user-attachments/assets/d93765a9-1df1-466c-8f94-105f97b6818e" />

>>>>>>> 8e665caf898e29312c928bdf80d0c93460ccf648
