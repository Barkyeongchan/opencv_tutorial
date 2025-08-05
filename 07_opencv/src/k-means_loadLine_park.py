'''
1. 이미지를 불러온다.
2. 평균 클러스터링을 사용해 색상을 분류한다.
3. 분류한 이미지를 출력한다.
'''

import cv2
import numpy as np
import matplotlib.pyplot as plt

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

# --- 색상 팔레트 생성 ---
palette = np.zeros((50, 300, 3), dtype=np.uint8)  # 가로 300px, 세로 50px
step = 300 // K
for i, color in enumerate(center):
    palette[:, i*step:(i+1)*step, :] = color

cv2.imshow('Color Palette', palette)

# --- 색상 분포 차트 및 상세 분석 ---

# 픽셀 수 계산
unique, counts = np.unique(label, return_counts=True)
total_pixels = data.shape[0]

# 클러스터 별 비율 계산
ratios = counts / total_pixels

# BGR → RGB 변환 (matplotlib는 RGB)
colors_rgb = center[:, ::-1] / 255.0  # 0~1 정규화

# 분포 차트 출력
plt.figure(figsize=(8, 4))
plt.bar(range(K), ratios, color=colors_rgb, tick_label=[f'C{i}' for i in range(K)])
plt.title('Cluster Color Distribution')
plt.xlabel('Cluster')
plt.ylabel('Pixel Ratio')
plt.ylim(0, 1)
plt.show()

# 상세 분석 출력
print("\n클러스터 상세 분석:")
for i in range(K):
    b, g, r = center[i]
    print(f"Cluster {i}: BGR=({b}, {g}, {r}), 픽셀 수={counts[i]}, 비율={ratios[i]:.4f}")

cv2.waitKey(0)
cv2.destroyAllWindows()