import easyocr
import cv2
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

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

# 이미지 출력
plt.figure(figsize=(10, 10))
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.axis('off')
plt.title('OCR 결과')
plt.show()