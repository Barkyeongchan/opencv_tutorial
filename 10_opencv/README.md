# 학습 목표 : 자신에게 맞는 머신러닝 학습법을 시도해본다.

# EasyORC 

## 목차

1. EasyORC

## 1. EasyORC

<details>
<summary></summary>
<div markdown="1">

## **1-1. EasyORC란?**

Python에서 **이미지로부터 텍스트를 추출 할 수 있게 해주는 오픈 소스 OCR 라이브러리**

**[설치 방법]**

```terminal
pip install easyocr
```

## **1-2. 중국어 교통 표지판 인식**

```python3
import easyocr
import cv2
import matplotlib.pyplot as plt

# 1. 이미지 불러오기 
reader = easyocr.Reader(['ch_tra', 'en'], gpu=False)
img_path = '../img/chinese_tra.jpg'
img = cv2.imread(img_path)

# 2. 이미지 텍스트 인식 
result = reader.readtext(img_path)
print(result)

# 3. 인식된 텍스트 확인해보기 

THRESHOLD = 0.5

for bbox, text, conf in result:
    if conf >= THRESHOLD:
        print(text)
        cv2.rectangle(img, pt1=bbox[0], pt2=bbox[2], color=(0, 255, 0), thickness=2)
    plt.figure(figsize=(8,8))
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.show()

cv2.waitKey(0)
cv2.destroyAllWindows()
```

**[출력 결과]**

<img width="512" height="327" alt="image" src="https://github.com/user-attachments/assets/08eda18b-8b9e-49bd-9d13-6e1d854e2c4c" />

<img width="801" height="866" alt="image" src="https://github.com/user-attachments/assets/a81a65d3-4555-4562-8e77-097b0cf7d3d5" />

<img width="106" height="121" alt="image" src="https://github.com/user-attachments/assets/a943e67a-cce3-43c8-99d8-cd2fea17cd98" />

## **1-3. 한국 교통 표지판 인식**

```python3
import easyocr
import cv2
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

font_path = "C:/Windows/Fonts/batang.ttc"  # 바탕체
fontprop = fm.FontProperties(fname=font_path, size=14)
plt.rc('font', family=fontprop.get_name())

# EasyOCR 리더: 한글 + 영어 사용
reader = easyocr.Reader(['ko', 'en'], gpu=False)

# 교통 표지판 이미지 경로
img_path = '../img/ko_sign.png'  # ← 여기에 교통표지판 이미지 파일 이름

# 이미지 로드
img = cv2.imread(img_path)

# OCR 수행
results = reader.readtext(img)

# 결과 출력 및 시각화
for bbox, text, conf in results:
    print(f"[인식된 글자] {text} (신뢰도: {conf:.2f})")

    # 박스 그리기
    (top_left, top_right, bottom_right, bottom_left) = bbox
    top_left = tuple(map(int, top_left))
    bottom_right = tuple(map(int, bottom_right))

    cv2.rectangle(img, top_left, bottom_right, (0, 255, 0), 2)
    cv2.putText(img, text, (top_left[0], top_left[1] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

# 이미지 출력
plt.figure(figsize=(10, 10))
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.axis('off')
plt.title('OCR 결과')
plt.show()
```

**[출력 결과]**

<img width="476" height="329" alt="image" src="https://github.com/user-attachments/assets/9d1cb551-bb05-4332-bbf3-37a81ab716e9" />

<img width="789" height="568" alt="image" src="https://github.com/user-attachments/assets/4b958f53-138d-4793-85f8-601a00a38982" />


<img width="330" height="276" alt="image" src="https://github.com/user-attachments/assets/be41518e-a04d-416f-ae03-38ed65dcdd81" />

</div>
</details>

