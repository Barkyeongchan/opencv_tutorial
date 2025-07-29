import cv2
#넘파이 임포드 (배열 사용)
import numpy as np

image = cv2.imread('../img/like_lenna.png', cv2.IMREAD_GRAYSCALE)

# @이미지 리사이징(resize)
#cv2.imshow('Image Window', image_small)
#image_small = cv2.resize(image, (100, 100))

# @사이즈 선언
#new_height = 300
#new_width = 300

# @nupy를 사용해 이미지 사이즈 변경
#dst = np.zeros((new_height, new_width, 3), dtype=np.uint8)
#resized_image = cv2.resize(image, (new_width, new_height), dst=dst)

# @배율로 사이즈 변경
#image_big = cv2.resize(image,dsize=None, fx=2, fy=2) # 두배로 늘림

# @대칭 변환
#image_fliped = cv2.flip(image, 0) # 상하 반전
#image_fliped = cv2.flip(image, 1) # 좌우 반전

# @회전 변환
#height, width, _ = image.shape # image.shape = (224, 224, 3) 세자리 값이기 때문에 반환 위치에 _를 넣어 반환값을 무시함
#matrix = cv2.getRotationMatrix2D((width/2, height/2), 90, 1)
#result = cv2.warpAffine(image, matrix, (width, height))
#matrix = cv2.getRotationMatrix2D((width/2,height/2),30,1)
#result = cv2.warpAffine(image,matrix,(width,height),borderValue=200)

# @이미지 자르기
cuted_image = image[50:150,50:150].copy()
cuted_image[:] = 200

# @이미지를 보여주는 명령어
cv2.imshow('Image Window', cuted_image)

# @이미지 크기 출력(shape)
print(cuted_image.shape)

cv2.waitKey(0)
cv2.destroyAllWindows()