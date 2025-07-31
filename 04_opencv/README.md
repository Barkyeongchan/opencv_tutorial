# 이미지 제어와 이미지 뒤틀기 / 영상 필터와 블러링 / 경계 검출

## 목차
1. 이미지 제어
2. 이미지 뒤틀기
3. 개인 프로젝트 (자동차 번호팜 추출)
4. 영상 필터와 컨볼루션
5. 블러링
6. 경계 검출

## 1. 이미지 제어

1. **이미지 이동(Translation)**

**이미지 이동이란?**

원래 있던 좌표에 이동하려는 거리만큼 더하여 이미지를 이동시키는 방법

```
x_new = x_old + d₁
y_new = y_old + d₂
```

아래와 같은 함수를 갖는다

```
dst = cv2.warpAffine(src, matrix, dsize, dst, flags, borderMode, borderValue)

src: 원본 이미지, numpy 배열
matrix: 2 x 3 변환행렬, dtype=float32
dsize: 결과 이미지의 크기, (width, height)
flags(optional): 보간법 알고리즘 플래그
borderMode(optional): 외곽 영역 보정 플래그
borderValue(optional): cv2.BORDER_CONSTANT 외곽 영역 보정 플래그일 경우 사용할 색상 값 (default=0)
dst: 결과 이미지
```

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

```

2. **이미지 확대/축소(Scaling)**

**이미지 확대/축소란?**

원래 있던 좌표에 이동 하려는 거리만큼 곱한다

```
x_new = a₁ * x_old
y_new = a₂ * y_old
```

> _보간법_

3. **이미지 회전(Rotation)**

**이미지 회전을 위한 변환 행렬식**

> _호도법_

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