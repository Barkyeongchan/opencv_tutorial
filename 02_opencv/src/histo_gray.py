import cv2
import numpy as np
import matplotlib.pylab as plt

# @이미지를 그레이스케일로 읽고 출력
img = cv2.imread('../img/like_lenna.png', cv2.IMREAD_GRAYSCALE)
cv2.imshow('img', img)

# @히스토그램 계산 및 그리기
hist = cv2.calcHist([img], [0], None, [256], [0, 256])
plt.plot(hist)

print("hist.shape : ", hist.shape)      # 히스토그램의 shapr 값
print("hist.sum() : ", hist.sum(), "img.shape : ", img.shape)   # 히스토그램 총 합계와 이미지 크기 
plt.show()