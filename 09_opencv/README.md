# [학습 목표 : OpenCV 머신러닝을 활용해 얼굴을 구별하고 표정을 구별 할 수 있다.]

# 하르 캐스케이드 분류기 (Haarcascade) / 

## 목차

1. 하르 캐스케이드 분류기
   - 하르 케스케이드 분류기란?
   - 하르 캐스케이드 얼굴 검출 예시

## 1. 하르 캐스케이드 분류기 (Haarcascade)

<details>
<summary></summary>
<div markdown="1">

## **1-1. 하르 캐스케이드 분류기란?**

개발자가 **직접 머신러닝 학습 알고리즘을 사용하지 않고도 객체를 검출**할 수 있도록 OpenCV가 제공하는 대표적인 상위 레벨 AP

openCV에서는 `[하르 케스케이드 xml](https://github.com/opencv/opencv/tree/master/data/haarcascades)` 형태로 제공한다.

cv2.CascadeClassifier([filename]) 와 classifier.detectMultiScale(img, scaleFactor, minNeighbors , flags, minSize, maxSize) 함수를 사용한다.

```
classifier = cv2.CascadeClassifier([filename]): 케스케이드 분류기 생성자
```

`filename` : 검출기 저장 파일 경로
`classifier` : 캐스케이드 분류기 객체

```
rect = classifier.detectMultiScale(img, scaleFactor, minNeighbors , flags, minSize, maxSize)
```

`img` : 입력 이미지
`scaleFactor` : 이미지 확대 크기에 제한. 1.3~1.5 (큰값: 인식 기회 증가, 속도 감소)
`minNeighbors` : 요구되는 이웃 수(큰 값: 품질 증가, 검출 개수 감소)
`flags` : 지금 사용안함
`minSize, maxSize` : 해당 사이즈 영역을 넘으면 검출 무시
`rect` : 검출된 영역 좌표 (x, y, w, h)

## **1-2. 하르 캐스케이드 얼굴 검출 예시**

<img width="1134" height="756" alt="image" src="https://github.com/user-attachments/assets/f4e2aba9-a50e-4f53-a7be-5cc7bf5dc263" />

<img width="755" height="500" alt="image" src="https://github.com/user-attachments/assets/51bcbe0a-d8c3-43a1-8621-48722b059d60" />

## **1-3. 캐스케이드 분류기로 얼굴과 눈 검출 실습**

**[1. 코드 생성]**

```python3
import numpy as np
import cv2

# 얼굴 검출을 위한 케스케이드 분류기 생성
face_cascade = cv2.CascadeClassifier('./data/haarcascade_frontalface_default.xml')

# 눈 검출을 위한 케스케이드 분류기 생성
eye_cascade = cv2.CascadeClassifier('./data/haarcascade_eye.xml')

# 검출할 이미지 읽고 그레이 스케일로 변환
img = cv2.imread('../img/children.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# 얼굴 검출
faces = face_cascade.detectMultiScale(gray)

# 검출된 얼굴 순회
for (x,y,w,h) in faces:
    # 검출된 얼굴에 사각형 표시
    cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
    # 얼굴 영역을 ROI로 설정
    roi = gray[y:y+h, x:x+w]
    # ROI에서 눈 검출
    eyes = eye_cascade.detectMultiScale(roi)
    # 검출된 눈에 사각형 표
    for (ex,ey,ew,eh) in eyes:
        cv2.rectangle(img[y:y+h, x:x+w],(ex,ey),(ex+ew,ey+eh),(0,255,0),2)

# 결과 출력 
cv2.imshow('img',img)
cv2.waitKey(0)
cv2.destroyAllWindows()
```

**[2. haarcascade_frontalface_default.xml / haarcascade_eye.xml 다운로드]**

[openCV github](https://github.com/opencv/opencv/tree/master/data/haarcascades)

**[3. 코드 실행 결과]**

<img width="510" height="525" alt="image" src="https://github.com/user-attachments/assets/fa58dd80-2660-4e25-9c71-4cd3a0c44f46" />


</div>
</details>
