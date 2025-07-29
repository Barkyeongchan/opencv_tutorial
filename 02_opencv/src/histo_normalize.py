import cv2
import numpy as np
import matplotlib.pylab as plt

# @이미지 그레이스케일로 불러오기
img = cv2.imread('../img/like_lenna.png', cv2.IMREAD_GRAYSCALE)

# @직접 정규화 게산
img_f = img.astype(np.float32)
img_norm = ((img_f - img_f.min()) * (255) / (img_f.max() - img_f.min()))
img_norm = img_norm.astype(np.uint8)

# @히스토그램 계산
hist = cv2.calcHist([img], [0], None, [256], [0, 255])

cv2.imshow('Before', img)

hists = {'Before' : hist}
for i, (k, v) in enumerate(hists.items()):
    plt.subplot(1,3,i+1)
    plt.title(k)
    plt.plot(v)
plt.show()