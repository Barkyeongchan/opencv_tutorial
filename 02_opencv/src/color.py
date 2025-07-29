import cv2
import numpy as np

# 기본값
img = cv2.imread('../img/like_lenna.png')

# BGR - openCV는 B, G, R 순서대로 표기함
bgr = cv2.imread('../img/like_lenna.png', cv2.IMREAD_COLOR)

# a
bgra = cv2.imread('../img/like_lenna.png', cv2.IMREAD_UNCHANGED)

# shape
print("default", img.shape, "color", bgr.shape, "unchanged", bgra.shape)

cv2.imshow('img', img)
cv2.imshow('bgr', bgr)
cv2.imshow('bgra', bgra)
cv2.imshow('alpha', bgra[:,:,3]) # 알파 채널만 표시

cv2.waitKey(0)
cv2.destroyAllWindows()