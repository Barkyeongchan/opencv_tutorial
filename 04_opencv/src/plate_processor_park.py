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


# @추출된 번호판 이미지 대비 최대화 함수
def maximize_contrast(gray_plate):
    # 모폴로지 연산용 구조화 요소 (번호판용으로 작게 설정)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))  # 3x3 → 2x2로 축소
    
    # Top Hat: 밝은 세부사항 (흰 배경) 강조
    tophat = cv2.morphologyEx(gray_plate, cv2.MORPH_TOPHAT, kernel)
    
    # Black Hat: 어두운 세부사항 (검은 글자) 강조  
    blackhat = cv2.morphologyEx(gray_plate, cv2.MORPH_BLACKHAT, kernel) 
    
    # 대비 향상 적용
    enhanced = cv2.add(gray_plate, tophat)
    enhanced = cv2.subtract(enhanced, blackhat)

    # 추가: 히스토그램 균등화로 대비 더욱 향상
    enhanced = cv2.equalizeHist(enhanced)   

    # 결과 비교
    plt.figure(figsize=(15, 4))

    plt.subplot(1, 4, 1)        # 1행 4열의 배열에서 첫 번째 위치
    plt.imshow(gray_plate, cmap='gray') # 그레이스케일된 이미지
    plt.title('Original Gray')  # 서브플롯 제목
    plt.axis('off')             # 축 눈금, 테두리 제거

    plt.subplot(1, 4, 2)
    plt.imshow(tophat, cmap='gray')     # 탑햇
    plt.title('Top Hat')
    plt.axis('off')

    plt.subplot(1, 4, 3)
    plt.imshow(blackhat, cmap='gray')   # 블랫햇
    plt.title('Black Hat')
    plt.axis('off')

    plt.subplot(1, 4, 4)
    plt.imshow(enhanced, cmap='gray')   # 대비 향상된 이미지 
    plt.title('Enhanced Contrast')
    plt.axis('off')

    plt.tight_layout()
    plt.show()

    return enhanced


# @대비 향상 기법 추가 함수 
def advanced_contrast_enhancement(gray_plate): 

    # CLAHE (적응형 히스토그램 균등화)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(2,1))  # 번호판용 설정
    clahe_result = clahe.apply(gray_plate)

    return clahe_result


# @번호판 적응형 임계처리 함수
def adaptive_threshold_plate(enhanced_plate):

    # 1단계: 가벼운 블러링 (노이즈 제거, 글자는 보존)
    blurred = cv2.GaussianBlur(enhanced_plate, (5, 5), 0)  # 5x5 → 3x3로 축소

    # 2단계: 번호판 최적화 적응형 임계처리
    thresh_adaptive = cv2.adaptiveThreshold(

        blurred,
        maxValue=255,
        adaptiveMethod=cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        thresholdType=cv2.THRESH_BINARY,  # BINARY_INV 대신 BINARY 사용
        blockSize=21,  # 19 → 11로 축소 (번호판 크기에 맞춤)
        C=5           # 9 → 2로 축소 (세밀한 조정)
    )

    # 3단계: Otsu 임계처리와 비교
    _, thresh_otsu = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # @직접 임계값 조정
    threshold_value = 35
    _, thresh_manual = cv2.threshold(blurred, threshold_value, 255, cv2.THRESH_BINARY_INV)

    # 4단계: 결과 비교
    plt.figure(figsize=(16, 4))   

    plt.subplot(1, 4, 1)
    plt.imshow(enhanced_plate, cmap='gray')     # 대비 향상 이미지
    plt.title('Enhanced Plate')
    plt.axis('off')

    plt.subplot(1, 4, 2)
    plt.imshow(thresh_manual, cmap='gray')      # 메뉴얼
    plt.title('manual')
    plt.axis('off')

    plt.subplot(1, 4, 3)
    plt.imshow(thresh_adaptive, cmap='gray')    # 적응형 쓰레시홀딩
    plt.title('Adaptive Threshold')
    plt.axis('off')

    plt.subplot(1, 4, 4)
    plt.imshow(thresh_otsu, cmap='gray')        # 오츠의 이진화 알고리즘
    plt.title('Otsu Threshold')
    plt.axis('off')

    plt.tight_layout()
    plt.show()

    return thresh_adaptive, thresh_otsu, thresh_manual


# @번호판에서 윤관선 검출하는 함수
def find_contours_in_plate(thresh_plate):
    
    # 윤곽선 검출
    contours, hierarchy = cv2.findContours(
        thresh_plate,                   # 이진화된 번호판 이미지
        mode=cv2.RETR_EXTERNAL,         # 가장 바깥쪽 윤곽선만 검출
        method=cv2.CHAIN_APPROX_SIMPLE  # 윤곽선 단순화
    )

    # 결과 시각화용 이미지 생성 (컬러)
    height, width = thresh_plate.shape
    contour_image = cv2.cvtColor(thresh_plate, cv2.COLOR_GRAY2BGR)


    # 모든 윤곽선을 다른 색으로 그리기
    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), 
              (255, 0, 255), (0, 255, 255), (128, 0, 128), (255, 165, 0)]

    for i, contour in enumerate(contours):
        color = colors[i % len(colors)]  # 색상 순환
        cv2.drawContours(contour_image, [contour], -1, color, 2)

        # 윤곽선 번호 표시
        M = cv2.moments(contour)

        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            cv2.putText(contour_image, str(i+1), (cx-5, cy+5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

    # 결과 시각화
    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    plt.imshow(thresh_plate, cmap='gray')
    plt.title('Binary Plate')
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.imshow(contour_image)
    plt.title(f'Contours Detected: {len(contours)}')
    plt.axis('off')

    # 윤곽선 정보 표시
    plt.subplot(1, 3, 3)
    contour_info = np.zeros((height, width, 3), dtype=np.uint8)

    for i, contour in enumerate(contours):
        area = cv2.contourArea(contour)
        x, y, w, h = cv2.boundingRect(contour)

        # 경계 사각형 그리기
        cv2.rectangle(contour_info, (x, y), (x+w, y+h), colors[i % len(colors)], 1)

        # 면적 정보 표시 (작은 글씨로)
        cv2.putText(contour_info, f'A:{int(area)}', (x, y-2), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.25, (255, 255, 255), 1)

    plt.imshow(contour_info)
    plt.title('Bounding Rectangles')
    plt.axis('off')

    plt.tight_layout()
    plt.show()

    # 윤곽선 정보 출력
    print("=== 윤곽선 검출 결과 ===")
    print(f"총 윤곽선 개수: {len(contours)}")

    for i, contour in enumerate(contours):
        area = cv2.contourArea(contour)
        x, y, w, h = cv2.boundingRect(contour)
        aspect_ratio = w / h if h > 0 else 0
        print(f"윤곽선 {i+1}: 면적={area:.0f}, 크기=({w}×{h}), 비율={aspect_ratio:.2f}")

    return contours, contour_image

# @무인이동체 시점에서 윤곽선 활용
def prepare_for_next_step(contours, contours_plate):
    print("=== 다음 단계 준비 ===")

    # 윤곽선이 충분히 검출되었는지 확인
    if len(contours) < 5:
        print("윤곽선이 적게 검출되었습니다. 전처리 단계를 재검토하세요.")
    elif len(contours) > 20:
        print("윤곽선이 너무 많이 검출되었습니다. 노이즈 제거가 필요할 수 있습니다.")
    else:
        print("적절한 수의 윤곽선이 검출되었습니다.")

    # 잠재적 글자 후보 개수 추정
    potential_chars = 0
    for contour in contours:
        area = cv2.contourArea(contour)
        if 30 < area < 2000:  # 글자 크기 범위 추정
            potential_chars += 1

    print(f"잠재적 글자 후보: {potential_chars}개")

    return potential_chars

# @처리된 이미지 저장 함수
def save_processed_results(plate_name, gray_plate, advanced_plate, enhanced_plate, thresh_manual, contours_image):

    # 저장 폴더 생성
    save_dir = '../processed_plates'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # 각 단계별 결과 저장
    cv2.imwrite(f'{save_dir}/{plate_name}_1_gray.png', gray_plate)
    cv2.imwrite(f'{save_dir}/{plate_name}_2_advanced.png', advanced_plate)  
    cv2.imwrite(f'{save_dir}/{plate_name}_3_enhanced.png', enhanced_plate)
    cv2.imwrite(f'{save_dir}/{plate_name}_4_threshold.png', thresh_manual)
    cv2.imwrite(f'{save_dir}/{plate_name}_5_contours.png', contours_image)

    print(f"처리 결과 저장 완료: {save_dir}/{plate_name}_*.png")

# @메인 실행

plate_name = 'plate_01'
plate_img = load_extracted_plate('plate_01')        # plate_01.png 불러옴
if plate_img is not None:
    gray_plate = convert_to_grayscale(plate_img)    # 그레이 스케일로 변환

enhanced_plate = maximize_contrast(gray_plate)      # 대비 향상

advanced_plate = advanced_contrast_enhancement(enhanced_plate)  # 대비 추가 향상

thresh_adaptive, thresh_otsu, thresh_manual = adaptive_threshold_plate(advanced_plate)  #스레시홀딩

contours, contours_image = find_contours_in_plate(thresh_manual)  #윤곽선 검출

potential_chars = prepare_for_next_step(contours, contours_image)

save_processed_results(plate_name = plate_name,
                       gray_plate = gray_plate,
                       advanced_plate = advanced_plate,
                       enhanced_plate = enhanced_plate,
                       thresh_manual = thresh_manual,
                       contours_image = contours_image)

cv2.waitKey(0)
cv2.destroyAllWindows()