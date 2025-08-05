'''
1. 이미지를 불러온다.
2. 평균 클러스터링을 사용해 색상을 분류한다.
3. 분류한 이미지를 출력한다.
'''

import cv2
import numpy as np

K = 8  # 군집화 갯수

img = cv2.imread('../img/load_line.jpg')
# 이미지 사이즈를 1/5로 줄임
img = cv2.resize(img, None, fx=0.2, fy=0.2, interpolation=cv2.INTER_AREA)

data = img.reshape((-1, 3)).astype(np.float32)

# 반복 중지 조건
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
# 10회 반복, 결과 확인 후 변경

# 평균 클러스터링 적용
ret, label, center = cv2.kmeans(data, K, None, criteria, 10, cv2.KMEANS_PP_CENTERS)

# 중심값을 정수형으로 변환

center = np.uint8(center)
print(center)

# 각 레이블에 해당하는 중심값으로 픽셀 값 선택
res = center[label.flatten()]

# 원본 영상의 형태로 변환
res = res.reshape((img.shape))

# 결과 출력
merged = np.hstack((img, res))
cv2.imshow('Load Line', merged)

cv2.waitKey(0)
cv2.destroyAllWindows()