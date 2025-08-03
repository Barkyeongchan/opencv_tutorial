# QR코드 스캔 / 아루코 마커 (Aruco Marker)

## 목차
1. 색상 표현 방식
2. 관심 영역 (ROI)

## 1. QR코드 스캔
<details>
<summary></summary>
<div markdown="1">

## **1-1. pyzbar**

**pyzbar란?**

**QR코드나 바코드를 이미지, 실시간 영상을 통해 인식하는 데 사용되는 python 라이브러리**

[설치 방법]
```bash
pip install pyzbar
```

## **1-2. QR코드 스캔 후 웹사이트로 이동하는 코드 만들기**

[1. 기본 코드 작성]
```python3
import cv2 
import matplotlib.pyplot as plt
import pyzbar.pyzbar as pyzbar  # pyzbar 실행

img = cv2.imread('../img/frame.png')  # QR 이미지 불러오기
plt.imshow(img)                       # 이미지를 맷플롯에서 출력
plt.show()

cv2.waitKey(0)
cv2.destroyAllWindows()
```

[2. 이미지 흑백(그레이스케일)으로 변환]
```python3
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)    # 이미지 그레이스케일로 불러오기

plt.imshow(gray, cmap='gray')   # 매트플롯에서 그레이로 정의 필요
```

[3. pyzbar 디코딩 추가]
```python3
# @디코딩(pyzbar)
decoded = pyzbar.decode(gray)
print(decoded)
```

[4. QR코드의 데이터와 형식 출력 추가]
```python3
# @QR코드의 데이터와 형식 출력
for d in decoded:
    print(d.data.decode('utf-8'))
    print(d.type)

    # @QR인식을 위한 사각형 그리기
    #cv2.rectangle(img, ())
```

[5. 인식된 QR코드의 테두리를 표시하는 사각형 그리기]
```python3
# @QR을 인식하는 사각형 그리기
cv2.rectangle(img, (d.rect[0], d.rect[1]), (d.rect[0] + d.rect[2], d.rect[1] + d.rect[3]),\
             (0, 255, 0), 20)
```

[6. 인식된 QR코드에 데이터와 형식 텍스트를 출력하기]
```python3
barcode_data = d.data.decode('utf-8')   
barcode_type = d.type

text = '%s (%s)' % (barcode_data, barcode_type) # 바코드 데이터와 형식을 text에 저장

# @QR에 글자 넣기
cv2.putText(img, text, (d.rect[0], d.rect[3] + 450), cv2.FONT_HERSHEY_SIMPLEX, 3,\
           (0, 0, 0), 5, cv2.LINE_AA)
```

[7. 카메라 캡쳐를 사용하여 QR코드 인식]
```python3
cap = cv2.VideoCapture(0)   # 비디오 캡쳐 활성화

# @이미지 캡쳐 조건 추가
while (cap.isOpened()):     # 카메라 캡쳐가 열려있으면 
    ret, img = cap.read()

    if not ret:
        continue

# @'q'입력시 창 닫힘
key = cv2.waitKey(1)
if key == ord('q'):
    break
```

[8. QR코드 인식 후 입력된 웹사이트로 이동하기]
```python3
import webbrowser  # 웹 사이트로 이동하는 라이브러이 설치

# @웹 사이트 이동 횟수 제한 조건
link_opened = False


 # @웹 사이트 한 번 만 열기
if not link_opened and barcode_data.startswith("http"):
    webbrowser.open(barcode_data)
    link_opened = True
```

[9. 최종 코드]
```python3
import cv2
import matplotlib.pyplot as plt
import pyzbar.pyzbar as pyzbar
import webbrowser

# @카메라 캡쳐 활성화
cap = cv2.VideoCapture(0)

# @웹 사이트 이동 횟수 제한 조건
link_opened = False

# @카메라 캡쳐 조건 추가
while (cap.isOpened()):     # 카메라 캡쳐가 열려있는 동안 
    ret, img = cap.read()

    if not ret:
        continue

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    decoded = pyzbar.decode(gray)

    # @QR코드의 데이터와 형식 출력
    for d in decoded:
        x, y, w, h = d.rect     # QR코드의 x, y, w, h 값은 d.rect에 저장됨
        barcode_data = d.data.decode('utf-8')
        barcode_type = d.type

        # @웹 사이트 한 번 만 열기
        if not link_opened and barcode_data.startswith("http"):
            webbrowser.open(barcode_data)
            link_opened = True

        text = '%s (%s)' % (barcode_data, barcode_type) # 바코드 데이터와 형식을 text에 저장

        # @QR을 인식하는 사각형 그리기
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        # @QR옆에 text 넣기
        cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)

    cv2.imshow('camera', img)

    key = cv2.waitKey(1)
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
```

</div>
</details>

# aruco marker

1. 캘러브레이션

2. 마커 
