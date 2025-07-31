# 평균 필터를 생성하여 블러 적용

import cv2
import numpy as np

img = cv2.imread('../img/paper.jpg')
#'''
## @5x5 평균 필터 커널 생성
#kernel = np.array([[0.04, 0.04, 0.04, 0.04, 0.04],
#                   [0.04, 0.04, 0.04, 0.04, 0.04],
#                   [0.04, 0.04, 0.04, 0.04, 0.04],
#                   [0.04, 0.04, 0.04, 0.04, 0.04],
#                   [0.04, 0.04, 0.04, 0.04, 0.04]])
#'''
## @5X5 평균 필터 커널 생성
#kernel = np.ones((5,5))/5**2
#
## @필터 적용
#blured = cv2.filter2D(img, -1, kernel)
#
## @이미지 출력
#cv2.imshow('origin', img)
#cv2.imshow('avrg blur', blured)

# @blur() 함수로 블러링
blur1 = cv2.blur(img, (10,10))
# @boxFilter() 함수로 블러링 적용
blur2 = cv2.boxFilter(img, -1, (10,10))

merged = np.hstack( (img, blur1, blur2))
cv2.imshow('blur', merged)

cv2.waitKey()
cv2.destroyAllWindows()