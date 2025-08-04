# 이미지 매칭 / 

## 목차
1. 이미지 매칭
 - 
2. 

## 1. 이미지 매칭 (Image Matching)
</details>
<sumarry></sumarry>
</div markdown = '1'>

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



## **1-3. 유클리드 거리 (Euclidian distance)와 해밍 거리(Hamming distance)

**두 이미지가 얼마나 비슷한지를 측정하는 방법 중 가장 대표적인 두 가지는 다음과 같다.**

## [유클리드 거리]

**두 값의 차이로 거리를 계산한다.**

예) 5를 각각 1과 7로 비교할 경우, 5와의 유클리드 거리 5-1 = 4와 7과의 유클리드 거리 7 - 5 = 2이므로 차이가 작은 7이 5와 더 유사하다.
