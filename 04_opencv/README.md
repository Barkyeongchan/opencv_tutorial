## 1. 이미지 이동

**이미지 이동이란?**

원래 있던 좌표에 이동 하려는 거리만큼 더한다.

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