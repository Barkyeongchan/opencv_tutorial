# 이미지 매칭 / 

## 목차
1. 이미지 매칭
 - 이미지 매칭이란?
 - 평균 해시 매칭(Average Hash Matching)
 - 유클리드 거리 (Euclidian distance)와 해밍 거리(Hamming distance)
 - 템플릿 매칭 (Template Matching)

2. 

## 1. 이미지 매칭 (Image Matching)
<details>
<sumarry></sumarry>
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
