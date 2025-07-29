import cv2
import numpy as np
import matplotlib.pylab as plt

# @이미지 가져오기
img = cv2.imread('../img/sample.jpg')

# @이미지 리사이징
img = cv2.resize(img, dsize=None, fx=0.2, fy=0.2)

# 이미지 크롭하기 위해 현재 사이즈 구함
h, w = img.shape[:2]
side = min(h, w)
center_x, center_y = w // 2, h // 2
x1 = center_x - side // 2
x2 = center_x + side // 2
y1 = center_y - side // 2
y2 = center_y + side // 2

# @크롭된 이미지
crop = img[y1:y2, x1:x2].copy()

# @이미지 꾸미기 (cropped만 사용)
bgra = cv2.cvtColor(crop, cv2.COLOR_BGR2BGRA)  # 알파채널 추가
gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)  # 흑백 변환

blk_size = 9
C = 5
adap = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                             cv2.THRESH_BINARY, blk_size, C)    

ret, thresh_cv_gray = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

ret, thresh_cv = cv2.threshold(crop, 127, 255, cv2.THRESH_BINARY)

# @결과 출력
imgs = {
    'Original': crop,
    'BGRA': bgra,
    'Gray Scale': gray,
    'Threshold' : thresh_cv,
    'Gray Scalr Threshold' : thresh_cv_gray,
    'Adaptive Threshold': adap
}

for i, (key, value) in enumerate(imgs.items()):
    plt.subplot(3, 2, i + 1)
    plt.title(key)
    # 컬러는 RGB로 바꿔서 보여주고, 흑백은 그대로
    if len(value.shape) == 3 and value.shape[2] == 3:
        plt.imshow(cv2.cvtColor(value, cv2.COLOR_BGR2RGB))
    elif len(value.shape) == 3 and value.shape[2] == 4:
        plt.imshow(cv2.cvtColor(value, cv2.COLOR_BGRA2RGBA))
    else:
        plt.imshow(value, 'gray')
    plt.xticks([]), plt.yticks([])

plt.tight_layout()
plt.show()