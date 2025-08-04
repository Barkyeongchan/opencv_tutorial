# 이미지 매칭 / 이미지 특징점과 검출기 / 특징 매칭 / 배경 제거

## 목차
1. 이미지 매칭
   - 이미지 매칭이란?
   - 평균 해시 매칭(Average Hash Matching)
   - 유클리드 거리 (Euclidian distance)와 해밍 거리(Hamming distance)
   - 템플릿 매칭 (Template Matching)

2. 이미지 특징점과 검출기
   - 이미지 특징점이란?
   - 해리스 코너 검출 (Harris Corner Detection)
   - 시-토마시 검출 (Shi & Tomasi Detection)
   - 특징점 검출기
   - 검출기 예제
  
3. 특징 매칭
   - 특징 매칭이란?
   - 카메라 캡쳐를 사용한 특징 매칭
  
4. 개인 프로젝트 (카메라 캡쳐를 사용한 바코드 특징 매칭)

5. 배경 제거

## 1. 이미지 매칭 (Image Matching)
<details>
<summary></summary>
<div markdown="1">

## **1-1. 이미지 매칭이란?**

서로 다른 **두 이미지를 비교해서 짝이 맞는 같은 형태의 객체**가 있는지 찾아내는 기술

의미있는 특징들을 **적절한 숫자로 변환 후 그 숫자를 비교하여 적합성을 판단**한다.

> 특징을 대표하는 숫자를 _특징 벡터_ 또는 _특징 디스크립터_ 라고 한다.

## **1-2. 평균 해시 매칭(Average Hash Matching)**

효과는 떨어지지만 구현이 아주 간단한 이미지 매칭 기법

특징 벡터를 평균값으로 구한다.

[특징 벡터 구하는 방법]

> 1. 이미지를 가로 세로 비율과 무관하게 특정한 크기로 축소합니다.
>
> 2. 픽셀 전체의 평균값을 구해서 각 픽셀의 값이 평균보다 작으면 0, 크면 1로 바꿉니다.
>
> 3. 0 또는 1로만 구성된 각 픽셀 값을 1행 1열로 변환합니다. (이는 한 개의 2진수 숫자로 볼 수 있습니다.)

```python3
# 권총을 평균 해시로 변환, 16X16 크기의 평균 해시를 가진다.

import cv2

# @영상 읽어서 그레이 스케일로 변환
img = cv2.imread('../img/pistol.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# @8x8 크기로 축소
gray = cv2.resize(gray, (16,16))

# @영상의 평균값 구하기
avg = gray.mean()

# @평균값을 기준으로 0과 1로 변환
bin = 1 * (gray > avg)
print(bin)

# @2진수 문자열을 16진수 문자열로 변환
dhash = []
for row in bin.tolist():
    s = ''.join([str(i) for i in row])
    dhash.append('%02x'%(int(s,2)))
dhash = ''.join(dhash)
print(dhash)

cv2.namedWindow('pistol', cv2.WINDOW_GUI_NORMAL)
cv2.imshow('pistol', img)
cv2.waitKey(0)
```
<img width="410" height="266" alt="image" src="https://github.com/user-attachments/assets/072499eb-fed2-427c-b5b2-e194448c7312" />

<img width="770" height="415" alt="image" src="https://github.com/user-attachments/assets/e895f9ef-c752-47bb-adfc-d445c43d52fd" />



## **1-3. 유클리드 거리 (Euclidian distance)와 해밍 거리(Hamming distance)**

*두 이미지가 얼마나 비슷한지를 측정하는 방법 중 가장 대표적인 두 가지는 다음과 같다.*

## [유클리드 거리]

**두 값의 차이로 거리를 계산한다.**

예) 5를 각각 1과 7로 비교할 경우
5와의 유클리드 거리 5-1 = 4와, 7과의 유클리드 거리 7 - 5 = 2이므로 차이가 작은 7이 5와 더 유사하다.

openCV에서는 cv2.norm() 함수를 사용한다.
```
distance = cv2.norm(src1, src2, cv2.NORM_L2)
```
`src1` :	첫 번째 입력 배열 (NumPy 벡터, 이미지 등)

`src2` :	두 번째 입력 배열 (크기/타입 동일해야 함)

`cv2.NORM_L2` :	유클리드 거리 방식 (L2 노름)

`distance` :	반환값 - 두 배열 간 유클리드 거리 (float)

## [해밍 거리]

**두 값의 길이가 같을 때 각 자릿 값이 다른 것이 몇개인지를 계산한다.**

예) 12345를 각각 12354와 92345로 비교할 경우
12354와 마지막 두자리가 다르므로 해밍 거리 = 2와, 92345와 처음 한자리가 다르므로 햄이 거리 = 1이므로 92345와 더 유사하다.

openCV에서는 cv2.norm() 함수를 사용한다.
```
distance = cv2.norm(src1, src2, cv2.NORM_HAMMING)
```
`src1` :	첫 번째 이진 시퀀스 (예: dtype=uint8의 배열)

`src2` :	두 번째 이진 시퀀스 (크기 같아야 함)

`cv2.NORM_HAMMING` :	해밍 거리 방식 (각 비트 비교)

`distance` :	반환값 - 두 배열 간 해밍 거리 (int)

```python3
# 사물 이미지 중에서 권총 이미지 찾기, 16X16 평균 해쉬 사용

import cv2
import numpy as np
import glob

img = cv2.imread('../img/pistol.jpg')
cv2.imshow('query', img)

# @비교할 영상들이 있는 경로
search_dir = '../img/101_ObjectCategories'

# @이미지를 16x16 크기의 평균 해쉬로 변환
def img2hash(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, (16, 16))
    avg = gray.mean()
    bi = 1 * (gray > avg)
    return bi

# @해밍 거리 측정 함수
def hamming_distance(a, b):
    a = a.reshape(1,-1)
    b = b.reshape(1,-1)
    # 같은 자리의 값이 서로 다른 것들의 합
    distance = (a !=b).sum()
    return distance

# @권총 영상의 해쉬 구하기
query_hash = img2hash(img)

# @이미지 데이타 셋 디렉토리의 모든 영상 파일 경로
img_path = glob.glob(search_dir+'/**/*.jpg')
for path in img_path:

    # 데이타 셋 영상 한개 읽어서 표시
    img = cv2.imread(path)
    cv2.imshow('searching...', img)
    cv2.waitKey(5)

    # 데이타 셋 영상 한개의 해시
    a_hash = img2hash(img)

    # 해밍 거리 산출
    dst = hamming_distance(query_hash, a_hash)

    # 해밍거리 25% 이내만 출력
    if dst/256 < 0.25: 
        print(path, dst/256)
        cv2.imshow(path, img)
        
cv2.destroyWindow('searching...')
cv2.waitKey(0)
cv2.destroyAllWindows()
```
<img width="1280" height="700" alt="image" src="https://github.com/user-attachments/assets/c53f2746-1d31-4eef-8fa5-9a65cffe7227" />



## **1-4. 템플릿 매칭 (Template Matching)**

특정 물체에 대한 이미지를 준비해 두고 **그 물체가 포함되어 있을 것이라 에상할 수 있는 이미지와 비교 매칭**하여 위치는 찾는 방법

cv2.matchTemplate() 함수를 사용한다.
```
result = cv2.matchTemplate(img, templ, method, result, mask)
```
`img `: 입력 이미지

`templ` : 템플릿 이미지

`method` : 매칭 메서드

(cv2.TM_SQDIFF: 제곱 차이 매칭, 완벽 매칭:0, 나쁜 매칭: 큰 값

cv2.TM_SQDIFF_NORMED: 제곱 차이 매칭의 정규화

cv2.TM_CCORR: 상관관계 매칭, 완벽 매칭: 큰 값, 나쁜 매칭: 0

cv2.TM_CCORR_NORMED: 상관관계 매칭의 정규화

cv2.TM_CCOEFF: 상관계수 매칭, 완벽 매칭:1, 나쁜 매칭: -1

cv2.TM_CCOEFF_NORMED: 상관계수 매칭의 정규화)

`result(optional)` : 매칭 결과, (W - w + 1) x (H - h + 1) 크기의 2차원 배열

`mask(optional)` : TM_SQDIFF, TM_CCORR_NORMED인 경우 사용할 마스크

cv2.matchTemplate() 함수를 사용하여 반환된 베열의 최소값 또는 최대값을 구할 때는 cv2.minMaxLoc() 함수를 사용한다.
```
minVal, maxVal, minLoc, maxLoc = cv2.minMaxLoc(src, mask)
```
`src` : 입력 1 채널 배열

`minVal, maxVal` : 배열 전체에서의 최소 값, 최대 값

`minLoc, maxLoc` : 최소 값과 최대 값의 좌표 (x, y)

```python3
# 템플릿 매칭으로 객체 위치 검출

import cv2
import numpy as np

# @입력이미지와 템플릿 이미지 읽기
img = cv2.imread('../img/figures.jpg')
template = cv2.imread('../img/taekwonv1.jpg')
th, tw = template.shape[:2]
cv2.imshow('template', template)

# @3가지 매칭 메서드 순회
methods = ['cv2.TM_CCOEFF_NORMED', 'cv2.TM_CCORR_NORMED', \
                                     'cv2.TM_SQDIFF_NORMED']
'''
출력 값
cv2.TM_CCOEFF_NORMED 0.5131933093070984
cv2.TM_CCORR_NORMED 0.9238022565841675
cv2.TM_SQDIFF_NORMED 0.17028295993804932
'''

for i, method_name in enumerate(methods):
    img_draw = img.copy()
    method = eval(method_name)

    # 템플릿 매칭
    res = cv2.matchTemplate(img, template, method)

    # 최솟값, 최댓값과 그 좌표 구하기
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
    print(method_name, min_val, max_val, min_loc, max_loc)

    # TM_SQDIFF의 경우 최솟값이 좋은 매칭, 나머지는 그 반대
    if method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
        top_left = min_loc
        match_val = min_val
    else:
        top_left = max_loc
        match_val = max_val

    # 매칭 좌표 구해서 사각형 표시      
    bottom_right = (top_left[0] + tw, top_left[1] + th)
    cv2.rectangle(img_draw, top_left, bottom_right, (0,0,255),2)

    # 매칭 포인트 표시
    cv2.putText(img_draw, str(match_val), top_left, \
                cv2.FONT_HERSHEY_PLAIN, 2,(0,255,0), 1, cv2.LINE_AA)
    cv2.imshow(method_name, img_draw)
    
cv2.waitKey(0)
cv2.destroyAllWindows()
```
<img width="1280" height="470" alt="image" src="https://github.com/user-attachments/assets/da7b2714-c98c-477d-a650-3bcfeffc02a5" />

</div>
</details>

## 2. 이미지 특징점 (Keypoints)과 검출기 (Keypoints detector)

<details>
<summary></summary>
<div markdown="1">

## **2-1. 이미지 특징점이란?**

말 그대로 **이미지에서 특징이 되는 부분**

이미지끼리 매칭시 각 이미지에서 특징이 되는 부분을 비교한다. 즉, 이미지 특징점은 이미지를 매칭 할 때 사용됨

키포인트(Keypoints)라고 하며, 주로 **물체의 모서리나 코너를 특징점으로 사용**한다.

## **2-2. 해리스 코너 검출 (Harris Corner Detection)**

소벨(Sobel) 미분을 사용해 경곗값을 검출하여 경곗값의 경사도 변화량을 측정하여

**변화량이 수직, 수평, 대각선 방향으로 크게 변화하는 것을 코너로 판단하는 방법**

즉, 꼭직점을 특징점으로 사용하여 물체의 특징을 구분한다.

<img width="840" height="359" alt="image" src="https://github.com/user-attachments/assets/4e1262da-5853-47e6-9ef1-01961b1863c6" />



cv2.cornerHarris() 함수를 사용한다.
```
dst = cv2.cornerHarris(src, blockSize, ksize, k, dst, borderType)
```
`src` : 입력 이미지, 그레이 스케일

`blockSize` : 이웃 픽셀 범위

`ksize` : 소벨 미분 필터 크기

`k(optional)` : 코너 검출 상수 (보토 0.04~0.06)

`dst(optional)` : 코너 검출 결과 (src와 같은 크기의 1 채널 배열, 변화량의 값, 지역 최대 값이 코너점을 의미)

`borderType(optional)` : 외곽 영역 보정 형식

```python3
# 해리스 코너 검출

import cv2
import numpy as np

img = cv2.imread('../img/house.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# @해리스 코너 검출
corner = cv2.cornerHarris(gray, 2, 3, 0.04)

# @변화량 결과의 최대값 10% 이상의 좌표 구하기
coord = np.where(corner > 0.1* corner.max())
coord = np.stack((coord[1], coord[0]), axis=-1)

# @코너 좌표에 동그리미 그리기
for x, y in coord:
    cv2.circle(img, (x,y), 5, (0,0,255), 1, cv2.LINE_AA)

# @변화량을 영상으로 표현하기 위해서 0~255로 정규화
corner_norm = cv2.normalize(corner, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)

# @화면에 출력
corner_norm = cv2.cvtColor(corner_norm, cv2.COLOR_GRAY2BGR)
merged = np.hstack((corner_norm, img))

cv2.imshow('Harris Corner', merged)
cv2.waitKey()
cv2.destroyAllWindows()
```
<img width="1062" height="350" alt="image" src="https://github.com/user-attachments/assets/2da5b165-b1df-4765-9fb4-fd5987f9488b" />



## **2-3. 시-토마시 검출 (Shi & Tomasi Detection)**

해리스 코너 검출을 개선한 알고리즘

cv2.goodFeaturesToTrack() 함수를 사용한다.
```
corners = cv2.goodFeaturesToTrack(img, maxCorners, qualityLevel, minDistance, corners, mask, blockSize, useHarrisDetector, k)
```
`img` : 입력 이미지

`maxCorners` : 얻고 싶은 코너의 개수, 강한 것 순으로

`qualityLevel` : 코너로 판단할 스레시홀드 값

`minDistance` : 코너 간 최소 거리

`mask(optional)` : 검출에 제외할 마스크

`blockSize(optional)=3` : 코너 주변 영역의 크기

`useHarrisDetector(optional)=False` : 코너 검출 방법 선택 (True: 해리스 코너 검출 방법, False: 시와 토마시 코너 검출 방법)

`k(optional)` : 해리스 코너 검출 방법에 사용할 k 계수

`corners` : 코너 검출 좌표 결과, N x 1 x 2 크기의 배열, 실수 값이므로 정수로 변형 필요

**useHarrisDetector 파라미터에 True를 전달하면 해리스 코너 검출**

**디폴트 값인 False를 전달하면 시와 토마시 코너 검출**

```python3
# 시와 토마시 코너 검출

import cv2
import numpy as np

img = cv2.imread('../img/house.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# @시-토마스의 코너 검출 메서드
corners = cv2.goodFeaturesToTrack(gray, 80, 0.01, 10)

# @실수 좌표를 정수 좌표로 변환
corners = np.int32(corners)

# @좌표에 동그라미 표시
for corner in corners:
    x, y = corner[0]
    cv2.circle(img, (x, y), 5, (0,0,255), 1, cv2.LINE_AA)

cv2.imshow('Corners', img)
cv2.waitKey()
cv2.destroyAllWindows()
```
<img width="531" height="350" alt="image" src="https://github.com/user-attachments/assets/85040fea-5ab9-459c-8341-9ed3f3c0e246" />



## **2-4. 특징점 검출기**

특징점 검출기의 반환 결과는 특징점의 좌표뿐 아니라 **다양한 정보도 함께 출력**한다.

detector.detect() 함수를 사용한다._(detector에 각 특징점 검출기 함수를 대입)_
```
keypoints = detector.detect(img, mask): 특징점 검출 함수
```
`img` : 입력 이미지

`mask(optional)` : 검출 제외 마스크

`keypoints` : 특징점 검출 결과 (KeyPoint의 리스트)

```
Keypoint: 특징점 정보를 담는 객체
```
`pt` : 특징점 좌표(x, y), float 타입으로 정수 변환 필요

`size` : 의미 있는 특징점 이웃의 반지름

`angle` : 특징점 방향 (시계방향, -1=의미 없음)

`response` : 특징점 반응 강도 (추출기에 따라 다름)

`octave` : 발견된 이미지 피라미드 계층

`class_id` : 특징점이 속한 객체 ID

특징점을 표시해주는 전용 함수 cv2.drawKeypoints()를 사용한다.
```
outImg = cv2.drawKeypoints(img, keypoints, outImg, color, flags)
```
`img` : 입력 이미지

`keypoints` : 표시할 특징점 리스트

`outImg` : 특징점이 그려진 결과 이미지

`color(optional)` : 표시할 색상 (default: 랜덤)

`flags(optional)' : 표시 방법

(cv2.DRAW_MATCHES_FLAGS_DEFAULT: 좌표 중심에 동그라미만 그림(default)

cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS : 동그라미의 크기를 size와 angle을 반영해서 그림)

## **2-5. 검출기 예제**

**[1. GFTTDetector]**
```python3
# GFTTDetector로 특징점 검출

import cv2
import numpy as np
 
img = cv2.imread("../img/house.jpg")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Good feature to trac 검출기 생성
gftt = cv2.GFTTDetector_create() 
# 특징점 검출
keypoints = gftt.detect(gray, None)
# 특징점 그리기
img_draw = cv2.drawKeypoints(img, keypoints, None)

# 결과 출력
cv2.imshow('GFTTDectector', img_draw)
cv2.waitKey(0)
cv2.destrolyAllWindows()
```
<img width="531" height="350" alt="image" src="https://github.com/user-attachments/assets/4f8587c4-220d-4e30-bafd-4edc91108bdf" />



**[2. FAST(Feature from Accelerated Segment Test)]**

미분 계산을 하지않고 픽셀 중심으로 원을 그려 코너로 판단함

<img width="550" height="263" alt="image" src="https://github.com/user-attachments/assets/0a0959b3-0638-459a-ba53-01b1fc725e6b" />



```python3
# FAST로 특징점 검출

import cv2
import numpy as np

img = cv2.imread('../img/house.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# FASt 특징 검출기 생성
fast = cv2.FastFeatureDetector_create(50)

# 특징점 검출
keypoints = fast.detect(gray, None)

# 특징점 그리기
img = cv2.drawKeypoints(img, keypoints, None)

# 결과 출력
cv2.imshow('FAST', img)
cv2.waitKey()
cv2.destroyAllWindows()
```
<img width="531" height="350" alt="image" src="https://github.com/user-attachments/assets/f5256f33-a410-4144-b8bf-cdf811954f8d" />



**[3. SimpleBlobDetector]**

이진 스케일로 연결된 픽셀 그룹, 자잘한 객체는 노이즈로 한단하고 일정 크기 이상의 큰 객체만 찾는 검출기

```python3
# SimpleBolbDetector 검출기

import cv2
import numpy as np
 
img = cv2.imread("../img/house.jpg")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# SimpleBlobDetector 생성
detector = cv2.SimpleBlobDetector_create()

# 키 포인트 검출
keypoints = detector.detect(gray)

# 키 포인트를 빨간색으로 표시
img = cv2.drawKeypoints(img, keypoints, None, (0,0,255),\
                flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
 
cv2.imshow("Blob", img)
cv2.waitKey(0)
```
<img width="531" height="350" alt="image" src="https://github.com/user-attachments/assets/4e910264-2a0e-4930-b2ea-e0f464e957c1" />



**[4. SimpleBlobDetector에 필터 옵션 추가]**
```python3
# 필터 옵션으로 생성한 SimpleBlobDetector 검출기

import cv2
import numpy as np
 
img = cv2.imread("../img/house.jpg")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# blob 검출 필터 파라미터 생성
params = cv2.SimpleBlobDetector_Params()

# 경계값 조정
params.minThreshold = 10
params.maxThreshold = 240
params.thresholdStep = 5

# 면적 필터 켜고 최소 값 지정
params.filterByArea = True
params.minArea = 200
  
# 컬러, 볼록 비율, 원형비율 필터 옵션 끄기
params.filterByColor = False
params.filterByConvexity = False
params.filterByInertia = False
params.filterByCircularity = False 

# 필터 파라미터로 blob 검출기 생성
detector = cv2.SimpleBlobDetector_create(params)

# 키 포인트 검출
keypoints = detector.detect(gray)

# 키 포인트 그리기
img_draw = cv2.drawKeypoints(img, keypoints, None, None,\
                     cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

cv2.imshow("Blob with Params", img_draw)
cv2.waitKey(0)
```
<img width="531" height="350" alt="image" src="https://github.com/user-attachments/assets/1272d5b1-089a-4b51-9f45-b8d140081da6" />

</div>
</details>

## 3. 특징 매칭 (Feature Matching)

<details>
<summary></summary>
<div markdown="1">

## **3-1. 특징 매칭이란?**

서로 다른 두 이미지에서 **특징점과 특징 디스크립터**를 비교해 비슷한 객체끼리 짝짖는 것

cv2.DescriptorMatcher_create() 함수를 사용한다.
```
matcher = cv2.DescriptorMatcher_create(matcherType): 매칭기 생성자
```
`matcherType` : 생성할 구현 클래스의 알고리즘

("BruteForce": NORM_L2를 사용하는 BFMatcher

"BruteForce-L1": NORM_L1을 사용하는 BFMatcher

"BruteForce-Hamming": NORM_HAMMING을 사용하는 BRMatcher

"BruteForce-Hamming(2)": NORM_HAMMING2를 사용하는 BFMatcher

"FlannBased": NORM_L2를 사용하는 FlannBasedMatcher)

---

cv2.DescriptorMatcher_create() 함수를 사용하여 생성된 특징 매칭기에서 두 개의 디스크립터를 비교하는 함수는 세 가지가 있다.

<br><br>

**[1. matcher.match()]**
```
matches: matcher.match(queryDescriptors, trainDescriptors, mask): 1개의 최적 매칭
```
`queryDescriptors` : 특징 디스크립터 배열, 매칭의 기준이 될 디스크립터

`trainDescriptors` : 특징 디스크립터 배열, 매칭의 대상이 될 디스크립터

`mask(optional)` : 매칭 진행 여부 마스크

`matches` : 매칭 결과, DMatch 객체의 리스트

<br><br>

**[2. matcher.knnMatch()]**
```
matches = matcher.knnMatch(queryDescriptors, trainDescriptors, k, mask, compactResult): k개의 가장 근접한 매칭
```
`k` : 매칭할 근접 이웃 개수
`compactResult(optional)` : True: 매칭이 없는 경우 매칭 결과에 불포함 (default=False)

<br><br>

**[3. matcher.radiusMatch()]**
```
matches = matcher.radiusMatch(queryDescriptors, trainDescriptors, maxDistance, mask, compactResult): maxDistance 이내의 거리 매칭
```
`maxDistance` : 매칭 대상 거리

<br><br>

위의 세 함수의 반환 결과는 DMatch 객체 리스트로 받는다.
```
DMatch: 매칭 결과를 표현하는 객체
```
`queryIdx` : queryDescriptors의 인덱스

`trainIdx` : trainDescriptors의 인덱스

`imgIdx` : trainDescriptor의 이미지 인덱스

`distance` : 유사도 거리

<br><br>

매칭 결과를 시작적으로 표현하기 위해서 두 이미지를 하나로 합친 후 매칭점끼리 선으로 연결하는 작업을 drawMatches() 함수로 할 수 있다.
```
cv2.drawMatches(img1, kp1, img2, kp2, matches, flags): 매칭점을 이미지에 표시
```
`img1, kp1` : queryDescriptor의 이미지와 특징점

`img2, kp2` : trainDescriptor의 이미지와 특징점

`matches` : 매칭 결과

`flags` : 매칭점 그리기 옵션 (cv2.DRAW_MATCHES_FLAGS_DEFAULT: 결과 이미지 새로 생성(default값)

cv2.DRAW_MATCHES_FLAGS_DRAW_OVER_OUTIMG: 결과 이미지 새로 생성 안 함

cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS: 특징점 크기와 방향도 그리기

cv2.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS: 한쪽만 있는 매칭 결과 그리기 제외)

## **3-2. 카메라 캡쳐를 사용한 특징 매칭**

```python3
import cv2, numpy as np

img1 = None
win_name = 'Camera Matching'
MIN_MATCH = 10

# ORB 검출기 생성
detector = cv2.ORB_create(1000)

# Flann 추출기 생성
FLANN_INDEX_LSH = 6
index_params = dict(algorithm=FLANN_INDEX_LSH,
                   table_number=6,
                   key_size=12,
                   multi_probe_level=1)
search_params = dict(checks=32)
matcher = cv2.FlannBasedMatcher(index_params, search_params)

# 카메라 캡쳐 연결 및 프레임 크기 축소
cap = cv2.VideoCapture(0)              
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

while cap.isOpened():       
    ret, frame = cap.read() 
    if not ret:
        break
        
    if img1 is None:  # 등록된 이미지 없음, 카메라 바이패스
        res = frame
    else:             # 등록된 이미지 있는 경우, 매칭 시작
        img2 = frame
        gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
        
        # 키포인트와 디스크립터 추출
        kp1, desc1 = detector.detectAndCompute(gray1, None)
        kp2, desc2 = detector.detectAndCompute(img2, None)
        
        # 디스크립터가 없으면 건너뛰기
        if desc1 is None or desc2 is None or len(desc1) < 2 or len(desc2) < 2:
            res = frame
        else:
            # k=2로 knnMatch
            matches = matcher.knnMatch(desc1, desc2, 2)
            
            # 이웃 거리의 75%로 좋은 매칭점 추출
            ratio = 0.75
            good_matches = []
            for match_pair in matches:
                if len(match_pair) == 2:
                    m, n = match_pair
                    if m.distance < n.distance * ratio:
                        good_matches.append(m)
            
            print('good matches:%d/%d' % (len(good_matches), len(matches)))
            
            # matchesMask 초기화를 None으로 설정
            matchesMask = None
            
            # 좋은 매칭점 최소 갯수 이상인 경우
            if len(good_matches) > MIN_MATCH: 
                # 좋은 매칭점으로 원본과 대상 영상의 좌표 구하기
                src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
                dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
                
                # 원근 변환 행렬 구하기
                mtrx, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
                
                if mtrx is not None:
                    accuracy = float(mask.sum()) / mask.size
                    print("accuracy: %d/%d(%.2f%%)" % (mask.sum(), mask.size, accuracy * 100))
                    
                    if mask.sum() > MIN_MATCH:  # 정상치 매칭점 최소 갯수 이상인 경우
                        # 마스크를 리스트로 변환 (정수형으로)
                        matchesMask = [int(x) for x in mask.ravel()]
                        
                        # 원본 영상 좌표로 원근 변환 후 영역 표시
                        h, w = img1.shape[:2]
                        pts = np.float32([[[0, 0]], [[0, h-1]], [[w-1, h-1]], [[w-1, 0]]])
                        dst = cv2.perspectiveTransform(pts, mtrx)
                        img2 = cv2.polylines(img2, [np.int32(dst)], True, (0, 255, 0), 3, cv2.LINE_AA)
            
            # 매칭점 그리기
            res = cv2.drawMatches(img1, kp1, img2, kp2, good_matches, None, 
                                matchColor=(0, 255, 0),
                                singlePointColor=None,
                                matchesMask=matchesMask,
                                flags=cv2.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS)
    
    # 결과 출력
    cv2.imshow(win_name, res)
    key = cv2.waitKey(1) & 0xFF
    
    if key == 27:    # Esc, 종료
        break          
    elif key == ord(' '):  # 스페이스바를 누르면 ROI로 img1 설정
        x, y, w, h = cv2.selectROI(win_name, frame, False)
        if w and h:
            img1 = frame[y:y+h, x:x+w]
            print("ROI 선택됨: (%d, %d, %d, %d)" % (x, y, w, h))

cap.release()                          
cv2.destroyAllWindows()
```
<img width="638" height="508" alt="image" src="https://github.com/user-attachments/assets/ab40d7f8-af8e-4332-b9a4-af848841dd9f" />

<br><br>

<img width="638" height="511" alt="image" src="https://github.com/user-attachments/assets/bfbb3400-353b-41d9-81b1-b1e452fc7a8e" />

<br><br>

<img width="774" height="508" alt="image" src="https://github.com/user-attachments/assets/b1e05947-1224-45b6-a3af-6e8ca730fcb3" />

</div>
</details>

## 4. 개인 프로젝트 ()

<details>
<summary></summary>
<div markdown="1">

```python3
'''
1. 사용자가 스페이스 바를 누름
2. ROI 선택
3. 참조 이미지 저장
4. 실시간 매칭 시작
'''

import cv2, numpy as np
import time     # @@@타임 라이브러리 추가

last_match_time = 0     # @@@초기 값 선언
is_matching = False     # @@@현재 매칭 상태

# @초기 설정
img1 = None                     # ROI로 선택할 이미지
win_name = 'Camera Matching'    # 윈도우 이름
MIN_MATCH = 10                  # 최소 매칭점 갯수 (이 값 이하면 매칭 실패로 간주)

# @ORB 검출기 생성
# ORB_craete() = 이미지에서 ()개의 특징점을 찾는 알고리즘
detector = cv2.ORB_create(1000)

# @Flann 추출기 생성
# 두 이미지의 특징점을 빠르게 매칭
FLANN_INDEX_LSH = 6
index_params = dict(algorithm=FLANN_INDEX_LSH,
                   table_number=6,
                   key_size=12,
                   multi_probe_level=1)
search_params = dict(checks=32)
matcher = cv2.FlannBasedMatcher(index_params, search_params)

# @카메라 캡쳐 연결 및 프레임 크기 축소
cap = cv2.VideoCapture(0)              
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

while cap.isOpened():         # 카메라가 계속 작동하는 동안 
    ret, frame = cap.read() 
    if not ret:
        break
        
    if img1 is None:  # 등록된 이미지 없음, 카메라 바이패스
        res = frame
    else:             # 등록된 이미지 있는 경우, 매칭 시작
        img2 = frame

        # [step 1]
        gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)  # 참조 이미지, 그레이스케일로 바꿈
        gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)  # 현재 카메라 프레임, 그레이스케일로 바꿈
        gray1 = cv2.GaussianBlur(gray1, (5,5), 0) # @@@ 가우시안으로 전처리
        gray2 = cv2.GaussianBlur(gray2, (5,5), 0) # @@@ 가우시안으로 전처리

        # [step 2]
        # @키포인트와 디스크립터 추출
        # kp : keypoints 특징점 위치정보
        # desc : 특징점의 특성을 숫자로 표현
        kp1, desc1 = detector.detectAndCompute(gray1, None) # 참조 이미지의 특징점
        kp2, desc2 = detector.detectAndCompute(img2, None)  # 현재 카메라 이미지의 특징점
        
        # @디스크립터가 없으면 건너뛰기
        if desc1 is None or desc2 is None or len(desc1) < 2 or len(desc2) < 2:
            res = frame
        else:
            # [step 3]
            # k=2로 knnMatch : 각 특징점마다 가장 유사한 2개의 후보를 찾음
            matches = matcher.knnMatch(desc1, desc2, 2)
            
            # [step 4]
            # @이웃 거리의 50%로 좋은 매칭점 추출
            ratio = 0.5
            good_matches = []
            for match_pair in matches:
                if len(match_pair) == 2:
                    m, n = match_pair # 1등, 2등
                    if m.distance < n.distance * ratio: # 1등이 2등보다 25% 좋으면 
                        good_matches.append(m) # 추가한다.
            
            print('good matches:%d/%d' % (len(good_matches), len(matches)))
            
            # @matchesMask 초기화를 None으로 설정
            matchesMask = None
            
            # @좋은 매칭점 최소 갯수 이상인 경우
            if len(good_matches) > MIN_MATCH: 

                # 좋은 매칭점으로 원본과 대상 영상의 좌표 구하기
                src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
                dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
                
                # 원근 변환 행렬 구하기
                # RANSAC : 잘못된 매칭점의 outline 제거
                mtrx, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
                
                if mtrx is not None:
                    accuracy = float(mask.sum()) / mask.size
                    print("accuracy: %d/%d(%.2f%%)" % (mask.sum(), mask.size, accuracy * 100))
                    
                    if mask.sum() > MIN_MATCH:  # 정상치 매칭점 최소 갯수 이상인 경우
                        # 마스크를 리스트로 변환 (정수형으로)
                        matchesMask = [int(x) for x in mask.ravel()]
                        
                        # 결과 시작화
                        # 원본 영상 좌표로 원근 변환 후 영역 표시
                        h, w = img1.shape[:2]
                        pts = np.float32([[[0, 0]], [[0, h-1]], [[w-1, h-1]], [[w-1, 0]]])
                        dst = cv2.perspectiveTransform(pts, mtrx)
                        img2 = cv2.polylines(img2, [np.int32(dst)], True, (0, 255, 0), 3, cv2.LINE_AA)

                        # @@@ MATCH! 텍스트 출력 조건 추가
                        if mask.size >= 40 and accuracy * 100 >= 95.0:
                            last_match_time = time.time()
                            is_matching = True
            
            # @매칭점 그리기
            res = cv2.drawMatches(img1, kp1, img2, kp2, good_matches, None, 
                                matchColor=(0, 255, 0),
                                singlePointColor=None,
                                matchesMask=matchesMask,
                                flags=cv2.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS)
 
            # @@@ MATCH! 글자 출력 2초 동안 유지
            if time.time() - last_match_time <= 2.0:  # 최근 2초 이내라면
                cv2.putText(res, 'MATCH!', (50, 70), cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0, 0, 255), 4, cv2.LINE_AA)
            else:
                cv2.putText(res, 'NOT MATCH', (50, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 0, 0), 3, cv2.LINE_AA)
   
    # @결과 출력
    cv2.imshow(win_name, res)
    key = cv2.waitKey(1) & 0xFF
    
    if key == 27:    # Esc, 종료
        break          
    elif key == ord(' '):  # 스페이스바를 누르면 ROI로 img1 설정
        x, y, w, h = cv2.selectROI(win_name, frame, False)
        if w and h:
            img1 = frame[y:y+h, x:x+w]
            print("ROI 선택됨: (%d, %d, %d, %d)" % (x, y, w, h))

cap.release()                          
cv2.destroyAllWindows()
```
<img width="639" height="508" alt="image" src="https://github.com/user-attachments/assets/80496cf1-f5f8-43e3-9535-fd06a4508dc6" />

<br><br>

<img width="638" height="508" alt="image" src="https://github.com/user-attachments/assets/0b7f378f-8c8b-4ad0-8745-5902caea5e80" />

<br><br>

<img width="1058" height="507" alt="image" src="https://github.com/user-attachments/assets/078d9410-5318-47fc-baff-068525719526" />

<br><br>

<img width="1058" height="508" alt="image" src="https://github.com/user-attachments/assets/1ea69aba-5b37-473f-9195-ef376b35656a" />

</div>
</details>

## 5. 배경 제거

<details>
<semmary></semmary>
<div markdown="1">

## **5-1. 객체 추적 (Object Tracking)**

**동영상에서 지속적으로 움직이는 객체를 찾는 방법**

배경 제거는 객체 추적의 다양한 방법 중 하나이다.

## **5-2. 배경 제거(Background Subtraction)**

객체를 명확하게 파악하기 위해서 객체를 포함하는 영상에서 **객체가 없는 배경 영상을 빼는 방법** 즉, 배경을 모두 제거하여 객체만 남기는 것

<img width="473" height="250" alt="image" src="https://github.com/user-attachments/assets/e163899f-1baa-418e-a93a-c8022a63e66f" />

<br><br>

cv2.bgsegm.createBackgroundSubtractorMOG() 함수를 사용한다.
```
cv2.bgsegm.createBackgroundSubtractorMOG(history, nmixtures, backgroundRatio, noiseSigma)
```
`history=200` : 히스토리 길이

`nmixtures=5` : 가우시안 믹스처의 개수

`backgroundRatio=0.7` : 배경 비율

`noiseSigma=0` : 노이즈 강도 (0=자동)

<br><br>

배경 제거 객체 인터페이스 함수는 다음 주 종류가 있다.

```
foregroundmask = backgroundsubtractor.apply(img, foregroundmask, learningRate)
```
`img` : 입력 영상

`foregroundmask` : 전경 마스크

`learningRate=-1` : 배경 훈련 속도(0~1, -1: 자동)

<br><br>

```
backgroundImage = backgroundsubtractor.getBackgroundImage(backgroundImage)
```
`backgroundImage` : 훈련용 배경 이미지

<br><br>

```python3
# BackgroundSubtractorMOG로 배경 제거

import numpy as np, cv2

cap = cv2.VideoCapture('../img/walking.avi')
fps = cap.get(cv2.CAP_PROP_FPS) # 프레임 수 구하기
delay = int(1000/fps)

# 배경 제거 객체 생성
# history : 과거 프레임의 객수, 배경을 학습하는데 얼마나 많으 프레임을 기억할지 정함
# varThreshold : 픽셀이 객체인지 배경인지 구분하는 기준
fgbg = cv2.createBackgroundSubtractorMOG2(50, 45, detectShadows=False)
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    # 배경 제거 마스크 계산 --- ②
    fgmask = fgbg.apply(frame)
    cv2.imshow('frame',frame)
    cv2.imshow('bgsub',fgmask)
    if cv2.waitKey(1) & 0xff == 27:
        break

cap.release()
cv2.destroyAllWindows()
```
<img width="600" height="241" alt="image" src="https://github.com/user-attachments/assets/9967e633-dfd9-4cd8-87b4-5da698d990e8" />

<br><br>

