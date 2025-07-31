# 자동차 번호판 추출

import cv2
import numpy as np
import os
from matplotlib import pyplot as plt

# @추출된 번호판 이미지 불러오기
def load_extracted_plate(plate_name):
    plate_path = f'../extracted_plates/{plate_name}.png'

    if os.path.exists(plate_path):
        plate_img = cv2.imread(plate_path)
        print(f"번호판 이미지 로드 완료: {plate_img.shape}")
        return plate_img

    else:
        print(f"파일을 찾을 수 없습니다: {plate_path}")
        return None

# @추출된 번호판을 그레이스케일로 변환
def convert_to_grayscale(plate_img):
    gray_plate = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)    # BGR을 그레이스케일로 변환
    plt.figure(figsize=(12, 4))    # 결과 비교 시각화
    plt.subplot(1, 2, 1)
    plt.imshow(cv2.cvtColor(plate_img, cv2.COLOR_BGR2RGB))
    plt.title('Original Extracted Plate')
    plt.axis('off')
    
    plt.subplot(1, 2, 2)
    plt.imshow(gray_plate, cmap='gray')
    plt.title('Grayscale Plate')
    plt.axis('off')
   
    plt.tight_layout()
    plt.show()

    return gray_plate

# @함수 사용
plate_img = load_extracted_plate('plate_01')  # plate_01.png 불러옴
if plate_img is not None:
    gray_plate = convert_to_grayscale(plate_img)

cv2.waitKey(0)
cv2.destroyAllWindows()