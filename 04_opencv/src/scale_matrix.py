# 이미지 확대, 축소

import cv2
import numpy as np

img = cv2.imread('../img/fish.jpg')
height, width = img.shape[:2]   # 영상 크기 정의

# @0.5배 축소 변환 행렬
m_small = np.float32([[0.5, 0, 0],
                       [0, 0.5,0]])  
# @2배 확대 변환 행렬
m_big = np.float32([[2, 0, 0],
                     [0, 2, 0]])  

# @보간법 적용 없이 확대 축소
dst1 = cv2.warpAffine(img, m_small, (int(height*0.5), int(width*0.5)))
dst2 = cv2.warpAffine(img, m_big, (int(height*2), int(width*2)))

# @보간법 적용한 확대 축소
dst3 = cv2.warpAffine(img, m_small, (int(height*0.5), int(width*0.5)), \
                        None, cv2.INTER_AREA)
dst4 = cv2.warpAffine(img, m_big, (int(height*2), int(width*2)), \
                        None, cv2.INTER_CUBIC)

# @cv.2resize() 함수를 사용해 확대 축소
# 크기 지정으로 축소
func1 = cv2.resize(img, (int(width*0.5), int(height*0.5)), \
                         interpolation=cv2.INTER_AREA)

# 배율 지정으로 확대
func2 = cv2.resize(img, None,  None, 2, 2, cv2.INTER_CUBIC)

# @이미지 출력
cv2.imshow("original", img)
cv2.imshow("small", dst1)
cv2.imshow("big", dst2)
cv2.imshow("small INTER_AREA", dst3)
cv2.imshow("big INTER_CUBIC", dst4)
cv2.imshow("use Function small", func1)
cv2.imshow("use Function big", func2)

cv2.waitKey(0)
cv2.destroyAllWindows()