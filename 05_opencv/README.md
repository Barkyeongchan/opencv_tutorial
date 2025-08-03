# QR코드 스캔 / 아루코 마커 (Aruco Marker)

## 목차
1. 색상 표현 방식
2. 관심 영역 (ROI)

## 1. QR코드 스캔
<details>
<summery></summery>
## **1-1.pyzbar**

**pyzbar란?

QR코드나 바코드를 이미지, 실시간 영상을 통해 인식하는 데 사용되는 python 라이브러리

[설치 방법]
```bash
pip install pyzbar
```

예제 코드
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


# aruco marker

1. 캘러브레이션

2. 마커 
