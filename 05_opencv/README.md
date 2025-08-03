# QR코드 스캔 / 아루코 마커 (ArUco Marker)

## 목차
1. QR코드 스캔
2. 아루코 마커

## 1. QR코드 스캔
<details>
<summary></summary>
<div markdown="1">

## **1-1. pyzbar**

**pyzbar란?**

**QR코드나 바코드를 이미지, 실시간 영상을 통해 인식하는 데 사용되는 python 라이브러리**

[설치 방법]
```bash
pip install pyzbar
```

## **1-2. QR코드 스캔 후 웹사이트로 이동하는 코드 만들기**

[1. 기본 코드 작성]
```python3
import cv2 
import matplotlib.pyplot as plt
import pyzbar.pyzbar as pyzbar  # pyzbar 실행

img = cv2.imread('../img/frame.png')  # QR 이미지 불러오기
plt.imshow(img)                       # 이미지를 맷플롯에서 출력
plt.show()

cv2.waitKey(0)
cv2.destroyAllWindows()
```

[2. 이미지 흑백(그레이스케일)으로 변환]
```python3
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)    # 이미지 그레이스케일로 불러오기

plt.imshow(gray, cmap='gray')   # 매트플롯에서 그레이로 정의 필요
```

[3. pyzbar 디코딩 추가]
```python3
# @디코딩(pyzbar)
decoded = pyzbar.decode(gray)
print(decoded)
```

[4. QR코드의 데이터와 형식 출력 추가]
```python3
# @QR코드의 데이터와 형식 출력
for d in decoded:
    print(d.data.decode('utf-8'))
    print(d.type)

    # @QR인식을 위한 사각형 그리기
    #cv2.rectangle(img, ())
```

[5. 인식된 QR코드의 테두리를 표시하는 사각형 그리기]
```python3
# @QR을 인식하는 사각형 그리기
cv2.rectangle(img, (d.rect[0], d.rect[1]), (d.rect[0] + d.rect[2], d.rect[1] + d.rect[3]),\
             (0, 255, 0), 20)
```

[6. 인식된 QR코드에 데이터와 형식 텍스트를 출력하기]
```python3
barcode_data = d.data.decode('utf-8')   
barcode_type = d.type

text = '%s (%s)' % (barcode_data, barcode_type) # 바코드 데이터와 형식을 text에 저장

# @QR에 글자 넣기
cv2.putText(img, text, (d.rect[0], d.rect[3] + 450), cv2.FONT_HERSHEY_SIMPLEX, 3,\
           (0, 0, 0), 5, cv2.LINE_AA)
```

[7. 카메라 캡쳐를 사용하여 QR코드 인식]
```python3
cap = cv2.VideoCapture(0)   # 비디오 캡쳐 활성화

# @이미지 캡쳐 조건 추가
while (cap.isOpened()):     # 카메라 캡쳐가 열려있으면 
    ret, img = cap.read()

    if not ret:
        continue

# @'q'입력시 창 닫힘
key = cv2.waitKey(1)
if key == ord('q'):
    break
```

[8. QR코드 인식 후 입력된 웹사이트로 이동하기]
```python3
import webbrowser  # 웹 사이트로 이동하는 라이브러이 설치

# @웹 사이트 이동 횟수 제한 조건
link_opened = False


 # @웹 사이트 한 번 만 열기
if not link_opened and barcode_data.startswith("http"):
    webbrowser.open(barcode_data)
    link_opened = True
```

[9. 최종 코드]
```python3
import cv2
import matplotlib.pyplot as plt
import pyzbar.pyzbar as pyzbar
import webbrowser

# @카메라 캡쳐 활성화
cap = cv2.VideoCapture(0)

# @웹 사이트 이동 횟수 제한 조건
link_opened = False

# @카메라 캡쳐 조건 추가
while (cap.isOpened()):     # 카메라 캡쳐가 열려있는 동안 
    ret, img = cap.read()

    if not ret:
        continue

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    decoded = pyzbar.decode(gray)

    # @QR코드의 데이터와 형식 출력
    for d in decoded:
        x, y, w, h = d.rect     # QR코드의 x, y, w, h 값은 d.rect에 저장됨
        barcode_data = d.data.decode('utf-8')
        barcode_type = d.type

        # @웹 사이트 한 번 만 열기
        if not link_opened and barcode_data.startswith("http"):
            webbrowser.open(barcode_data)
            link_opened = True

        text = '%s (%s)' % (barcode_data, barcode_type) # 바코드 데이터와 형식을 text에 저장

        # @QR을 인식하는 사각형 그리기
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        # @QR옆에 text 넣기
        cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)

    cv2.imshow('camera', img)

    key = cv2.waitKey(1)
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
```

</div>
</details>

## 2. 아루코 마커 (ArUco Marker)
<details>
<summary></summary>
<div markdown="1">
    
## **2-1. 아루코 마커란?**

컴퓨터 비전에서 자주 사용되는 **마커 기반 추적 시스템**으로 openCv에서 자세 추정, 로봇 네비게이션, 증강현실 등 다영하게 사용된다.

ArUco marker는 **정사각형의 흑백 코드 마커로 고유의 ID를 가진 2D 바코드이다.

<img width="414" height="350" alt="image" src="https://github.com/user-attachments/assets/0dea7850-021b-4bcb-9d33-6bb1404a458c" />



## **2-2. 아루코 마커를 사용해 거리에 따른 경고 메세지 출력 알고리즘 만들기

[1. 캘리브레이션을 위한 이미지 촬영]
```python3
import cv2
import datetime

# 비디오 캡처 객체 생성
cap = cv2.VideoCapture(0) 

while True:

    # 카메라로부터 프레임을 읽음
    ret, frame = cap.read()
    if not ret:
        print("프레임 X")  # 프레임 읽기 실패 시 메시지 출력
        break

    # 읽은 프레임을 화면에 표시
    cv2.imshow("Video", frame)

    # 키 입력을 기다림 (1ms 대기 후 다음 프레임으로 이동)
    key = cv2.waitKey(1) & 0xFF

    # 'a' 키가 눌리면 현재 프레임을 저장
    if key == ord('a'):
        # 파일 이름을 현재 날짜 및 시간으로 설정
        filename = datetime.datetime.now().strftime("../img/capture_%Y%m%d_%H%M%S.png")
        # 프레임을 이미지 파일로 저장
        cv2.imwrite(filename, frame)
        print(f"{filename}")  # 저장된 파일 이름 출력

    # 'q' 키가 눌리면 루프를 종료
    elif key == ord('q'):
        break

# 자원 해제 (카메라 및 모든 OpenCV 창 닫기)
cap.release()
cv2.destroyAllWindows()
```
<img width="640" height="480" alt="image" src="https://github.com/user-attachments/assets/0008e74c-916c-4a3b-899e-8825dd5177a1" />



[2. 촬영한 이미지를 활용해 켈리브레이션 실행하기]
```python3
import cv2
import numpy as np
import os
import glob
import pickle

def test_different_checkerboard_sizes(img_path):
    """다양한 체커보드 크기로 테스트해보는 함수"""
    img = cv2.imread(img_path)
    if img is None:
        return None
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # 일반적인 체커보드 크기들
    checkerboard_sizes = [
        (7, 10), (10, 7),   # 원래 설정
        (6, 9), (9, 6),     # 8x10 체커보드
        (5, 8), (8, 5),     # 6x9 체커보드
        (4, 7), (7, 4),     # 5x8 체커보드
        (6, 8), (8, 6),     # 7x9 체커보드
        (5, 7), (7, 5),     # 6x8 체커보드
        (4, 6), (6, 4),     # 5x7 체커보드
        (3, 5), (5, 3),     # 4x6 체커보드
    ]
    
    print(f"\n=== {os.path.basename(img_path)} 체커보드 크기 테스트 ===")
    
    successful_sizes = []
    
    for size in checkerboard_sizes:
        ret, corners = cv2.findChessboardCorners(gray, size, 
                                               cv2.CALIB_CB_ADAPTIVE_THRESH +
                                               cv2.CALIB_CB_FAST_CHECK +
                                               cv2.CALIB_CB_NORMALIZE_IMAGE)
        if ret:
            print(f"✓ {size} 크기로 체커보드 검출 성공!")
            successful_sizes.append(size)
        else:
            print(f"✗ {size} 크기로 검출 실패")
    
    return successful_sizes

def analyze_image_quality(img_path):
    """이미지 품질 분석 함수"""
    img = cv2.imread(img_path)
    if img is None:
        print(f"이미지를 읽을 수 없습니다: {img_path}")
        return
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    print(f"\n=== {os.path.basename(img_path)} 이미지 분석 ===")
    print(f"이미지 크기: {img.shape[1]} x {img.shape[0]}")
    print(f"평균 밝기: {np.mean(gray):.1f}")
    print(f"밝기 표준편차: {np.std(gray):.1f}")
    
    # 대비 분석
    contrast = gray.max() - gray.min()
    print(f"대비: {contrast}")
    
    # 블러 정도 분석 (Laplacian variance)
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    print(f"선명도 (높을수록 좋음): {laplacian_var:.1f}")
    
    if laplacian_var < 100:
        print("⚠️  이미지가 흐릿할 수 있습니다.")
    if contrast < 100:
        print("⚠️  이미지 대비가 낮습니다.")
    if np.mean(gray) < 50:
        print("⚠️  이미지가 너무 어둡습니다.")
    elif np.mean(gray) > 200:
        print("⚠️  이미지가 너무 밝습니다.")

def show_preprocessed_image(img_path, checkerboard_size=(7, 10)):
    """전처리된 이미지를 보여주는 함수"""
    img = cv2.imread(img_path)
    if img is None:
        return
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # 다양한 전처리 방법들
    # 1. 히스토그램 평활화
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    gray_clahe = clahe.apply(gray)
    
    # 2. 가우시안 블러
    gray_blur = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # 3. 이진화
    _, gray_thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # 4. 적응적 이진화
    gray_adaptive = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                        cv2.THRESH_BINARY, 11, 2)
    
    # 각각에 대해 체커보드 검출 시도
    methods = [
        ("Original", gray),
        ("CLAHE", gray_clahe),
        ("Gaussian Blur", gray_blur),
        ("Threshold", gray_thresh),
        ("Adaptive Threshold", gray_adaptive)
    ]
    
    print(f"\n=== {os.path.basename(img_path)} 전처리 방법별 테스트 ===")
    
    best_result = None
    best_method = None
    
    for method_name, processed_img in methods:
        ret, corners = cv2.findChessboardCorners(processed_img, checkerboard_size,
                                               cv2.CALIB_CB_ADAPTIVE_THRESH +
                                               cv2.CALIB_CB_FAST_CHECK +
                                               cv2.CALIB_CB_NORMALIZE_IMAGE)
        
        if ret:
            print(f"✓ {method_name}: 체커보드 검출 성공!")
            if best_result is None:
                best_result = (processed_img, corners)
                best_method = method_name
        else:
            print(f"✗ {method_name}: 체커보드 검출 실패")
    
    # 결과 시각화
    if best_result is not None:
        processed_img, corners = best_result
        result_img = cv2.cvtColor(processed_img, cv2.COLOR_GRAY2BGR)
        cv2.drawChessboardCorners(result_img, checkerboard_size, corners, True)
        
        # 이미지 크기 조정
        height, width = result_img.shape[:2]
        if height > 600 or width > 800:
            scale = min(600/height, 800/width)
            new_width = int(width * scale)
            new_height = int(height * scale)
            result_img = cv2.resize(result_img, (new_width, new_height))
        
        cv2.imshow(f'Best Result - {best_method}', result_img)
        cv2.waitKey(2000)
        cv2.destroyAllWindows()
    
    return best_result is not None

def calibrate_camera_flexible():
    """유연한 체커보드 검출을 위한 캘리브레이션 함수"""
    
    # 다양한 이미지 형식과 경로 시도
    image_paths = [
        '../img/*.png', '../img/*.jpg', '../img/*.jpeg',
        './img/*.png', './img/*.jpg', './img/*.jpeg',
        'img/*.png', 'img/*.jpg', 'img/*.jpeg',
        '*.png', '*.jpg', '*.jpeg'
    ]
    
    images = []
    for path_pattern in image_paths:
        found_images = glob.glob(path_pattern)
        if found_images:
            images.extend(found_images)
    
    images = list(set(images))
    
    if not images:
        print("체커보드 이미지를 찾을 수 없습니다!")
        return None
    
    print(f"총 {len(images)}개의 이미지를 발견했습니다.")
    
    # 첫 번째 이미지로 체커보드 크기 자동 감지
    print("\n=== 체커보드 크기 자동 감지 ===")
    first_image = images[0]
    successful_sizes = test_different_checkerboard_sizes(first_image)
    
    if not successful_sizes:
        print("첫 번째 이미지에서 체커보드를 찾을 수 없습니다.")
        print("이미지 품질을 분석합니다...")
        analyze_image_quality(first_image)
        
        # 전처리 방법 테스트
        print("다양한 전처리 방법을 테스트합니다...")
        if show_preprocessed_image(first_image):
            print("전처리를 통해 검출이 가능할 수 있습니다.")
        
        return None
    
    # 가장 많이 검출된 크기 선택
    CHECKERBOARD = successful_sizes[0]
    print(f"선택된 체커보드 크기: {CHECKERBOARD}")
    
    # 캘리브레이션 진행
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    objpoints = []
    imgpoints = []
    
    # 3D 점의 세계 좌표 정의
    objp = np.zeros((1, CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
    objp[0,:,:2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)
    
    successful_detections = 0
    
    for i, fname in enumerate(images):
        print(f"처리 중: {os.path.basename(fname)} ({i+1}/{len(images)})")
        
        img = cv2.imread(fname)
        if img is None:
            continue
        
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # 여러 전처리 방법 시도
        preprocessing_methods = [
            ("original", gray),
            ("clahe", cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8)).apply(gray)),
            ("blur", cv2.GaussianBlur(gray, (3, 3), 0))
        ]
        
        corners_found = False
        for method_name, processed_gray in preprocessing_methods:
            ret, corners = cv2.findChessboardCorners(processed_gray, CHECKERBOARD,
                                                   cv2.CALIB_CB_ADAPTIVE_THRESH +
                                                   cv2.CALIB_CB_FAST_CHECK +
                                                   cv2.CALIB_CB_NORMALIZE_IMAGE)
            
            if ret:
                objpoints.append(objp)
                corners2 = cv2.cornerSubPix(processed_gray, corners, (11,11), (-1,-1), criteria)
                imgpoints.append(corners2)
                successful_detections += 1
                corners_found = True
                
                print(f"  ✓ 체커보드 검출 성공 ({method_name})")
                
                # 결과 시각화
                img_corners = cv2.drawChessboardCorners(img.copy(), CHECKERBOARD, corners2, ret)
                height, width = img_corners.shape[:2]
                if height > 600 or width > 800:
                    scale = min(600/height, 800/width)
                    new_width = int(width * scale)
                    new_height = int(height * scale)
                    img_corners = cv2.resize(img_corners, (new_width, new_height))
                
                cv2.imshow('Checkerboard Detection', img_corners)
                cv2.waitKey(300)
                break
        
        if not corners_found:
            print(f"  ✗ 체커보드 검출 실패")
    
    cv2.destroyAllWindows()
    
    print(f"\n총 {successful_detections}개 이미지에서 체커보드 검출 성공")
    
    if successful_detections < 3:
        print("캘리브레이션을 위해서는 최소 3개 이상의 성공적인 검출이 필요합니다.")
        return None
    
    # 카메라 캘리브레이션 수행
    print("카메라 캘리브레이션을 수행 중...")
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints,
                                                      gray.shape[::-1], None, None)
    
    if ret:
        print("캘리브레이션 성공!")
        print("Camera matrix:")
        print(mtx)
        print("\nDistortion coefficients:")
        print(dist)
        
        calibration_data = {
            'camera_matrix': mtx,
            'dist_coeffs': dist,
            'rvecs': rvecs,
            'tvecs': tvecs,
            'checkerboard_size': CHECKERBOARD
        }
        
        with open('camera_calibration.pkl', 'wb') as f:
            pickle.dump(calibration_data, f)
        
        print("캘리브레이션 데이터가 저장되었습니다.")
        return calibration_data
    else:
        print("캘리브레이션 실패!")
        return None

def live_video_correction(calibration_data):
    """실시간 비디오 왜곡 보정"""
    if calibration_data is None:
        print("캘리브레이션 데이터가 없습니다.")
        return
    
    mtx = calibration_data['camera_matrix']
    dist = calibration_data['dist_coeffs']
    
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("웹캠을 열 수 없습니다.")
        return
    
    print("실시간 왜곡 보정을 시작합니다. 'q'를 눌러 종료하세요.")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        h, w = frame.shape[:2]
        newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w,h), 1, (w,h))
        dst = cv2.undistort(frame, mtx, dist, None, newcameramtx)
        
        x, y, w_roi, h_roi = roi
        if all(v > 0 for v in [x, y, w_roi, h_roi]):
            dst = dst[y:y+h_roi, x:x+w_roi]
        
        try:
            original = cv2.resize(frame, (640, 480))
            corrected = cv2.resize(dst, (640, 480))
            combined = np.hstack((original, corrected))
            
            cv2.putText(combined, "Original", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.putText(combined, "Corrected", (650, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            
            cv2.imshow('Camera Calibration Result', combined)
        except:
            cv2.imshow('Original', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    print("=== 향상된 카메라 캘리브레이션 프로그램 ===")
    
    if os.path.exists('camera_calibration.pkl'):
        choice = input("기존 캘리브레이션 데이터를 사용하시겠습니까? (y/n): ")
        if choice.lower() == 'y':
            with open('camera_calibration.pkl', 'rb') as f:
                calibration_data = pickle.load(f)
        else:
            calibration_data = calibrate_camera_flexible()
    else:
        calibration_data = calibrate_camera_flexible()
    
    if calibration_data is not None:
        print("\n실시간 비디오 보정을 시작합니다...")
        live_video_correction(calibration_data)
    else:
        print("\n캘리브레이션에 실패했습니다.")
        print("다음 사항을 확인해보세요:")
        print("1. 체커보드가 명확하게 보이는 이미지인지 확인")
        print("2. 체커보드의 모든 코너가 이미지 안에 포함되어 있는지 확인")
        print("3. 이미지가 너무 흐리거나 어둡지 않은지 확인")
        print("4. 다양한 각도에서 촬영된 이미지들인지 확인")
```

[3. 아루코 마커 인식 시키기]
```python3
import cv2
import numpy as np
import os
import time
import pickle


def estimate_pose_single_marker(corners, marker_size, camera_matrix, dist_coeffs):
    """
    단일 마커의 포즈를 추정하는 함수 (OpenCV 4.7+ 호환)
    cv2.aruco.estimatePoseSingleMarkers의 대체 함수
    """
    # 마커의 3D 좌표 정의 (마커 중심을 원점으로)
    half_size = marker_size / 2
    object_points = np.array([
        [-half_size, half_size, 0],   
        [half_size, half_size, 0],    
        [half_size, -half_size, 0],   
        [-half_size, -half_size, 0]   
    ], dtype=np.float32)
    
    # 이미지 좌표 (2D)
    image_points = corners[0].astype(np.float32)
    
    # PnP 문제 해결
    success, rvec, tvec = cv2.solvePnP(
        object_points, image_points, camera_matrix, dist_coeffs
    )
    
    if success:
        return rvec, tvec
    else:
        return None, None


def live_aruco_detection(calibration_data):
    """
    실시간으로 비디오를 받아 ArUco 마커를 검출하고 3D 포즈를 추정하는 함수

    Args:
        calibration_data: 카메라 캘리브레이션 데이터를 포함한 딕셔너리
            - camera_matrix: 카메라 내부 파라미터 행렬
            - dist_coeffs: 왜곡 계수
    """
    # 캘리브레이션 데이터 추출
    camera_matrix = calibration_data['camera_matrix']
    dist_coeffs = calibration_data['dist_coeffs']

    # ArUco 검출기 설정
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_100)
    aruco_params = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(aruco_dict, aruco_params)

    # 마커 크기 설정 (미터 단위)
    marker_size = 0.05  # 예: 5cm = 0.05m

    # 카메라 설정
    cap = cv2.VideoCapture(0)

    # 카메라 초기화 대기
    time.sleep(2)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break

        # 이미지 왜곡 보정
        frame_undistorted = cv2.undistort(frame, camera_matrix, dist_coeffs)

        # 마커 검출
        corners, ids, rejected = detector.detectMarkers(frame_undistorted)

        # 마커가 검출되면 표시 및 포즈 추정
        if ids is not None:
            # 검출된 마커 표시
            cv2.aruco.drawDetectedMarkers(frame_undistorted, corners, ids)

            # 각 마커에 대해 처리
            for i in range(len(ids)):
                # 포즈 추정 (새로운 방법으로 대체)
                rvec, tvec = estimate_pose_single_marker(
                    [corners[i]], marker_size, camera_matrix, dist_coeffs
                )
                
                if rvec is not None and tvec is not None:
                    # 좌표축 표시
                    cv2.drawFrameAxes(frame_undistorted, camera_matrix, dist_coeffs,
                                      rvec, tvec, marker_size/2)

                    # 마커의 3D 위치 표시
                    pos_x = tvec[0][0]
                    pos_y = tvec[1][0]
                    pos_z = tvec[2][0]

                    # 회전 벡터를 오일러 각도로 변환
                    rot_matrix, _ = cv2.Rodrigues(rvec)
                    euler_angles = cv2.RQDecomp3x3(rot_matrix)[0]

                    # 마커 정보 표시
                    corner = corners[i][0]
                    center_x = int(np.mean(corner[:, 0]))
                    center_y = int(np.mean(corner[:, 1]))

                    cv2.putText(frame_undistorted,
                                f"ID: {ids[i][0]}",
                                (center_x, center_y - 40),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.5, (255, 0, 255), 2)

                    cv2.putText(frame_undistorted,
                                f"Pos: ({pos_x:.2f}, {pos_y:.2f}, {pos_z:.2f})m",
                                (center_x, center_y),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.5, (255, 0, 255), 2)

                    cv2.putText(frame_undistorted,
                                f"Rot: ({euler_angles[0]:.1f}, {euler_angles[1]:.1f}, {euler_angles[2]:.1f})deg",
                                (center_x, center_y + 20),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.5, (255, 0, 255), 2)
                    
                    # 코너 포인트 표시
                    for point in corner:
                        x, y = int(point[0]), int(point[1])
                        cv2.circle(frame_undistorted, (x, y), 4, (0, 0, 255), -1)

        # 프레임 표시
        cv2.imshow('ArUco Marker Detection', frame_undistorted)

        # 'q' 키를 누르면 종료
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # 리소스 해제
    cap.release()
    cv2.destroyAllWindows()


def main():
    # 캘리브레이션 데이터 로드
    try:
        with open('camera_calibration.pkl', 'rb') as f:
            calibration_data = pickle.load(f)
        print("Calibration data loaded successfully")
    except FileNotFoundError:
        print("Error: Camera calibration file not found")
        return
    except Exception as e:
        print(f"Error loading calibration data: {e}")
        return

    print("Starting ArUco marker detection...")
    live_aruco_detection(calibration_data)


if __name__ == "__main__":
    main()
```

[4. 거리에 따른 경고 메세지 출력 알고리즘 적용]
```python3
import cv2
import numpy as np
import os
import time
import pickle


def estimate_pose_single_marker(corners, marker_size, camera_matrix, dist_coeffs):
    """
    단일 마커의 포즈를 추정하는 함수 (OpenCV 4.7+ 호환)
    cv2.aruco.estimatePoseSingleMarkers의 대체 함수
    """
    # 마커의 3D 좌표 정의 (마커 중심을 원점으로)
    half_size = marker_size / 2
    object_points = np.array([
        [-half_size, half_size, 0],   
        [half_size, half_size, 0],    
        [half_size, -half_size, 0],   
        [-half_size, -half_size, 0]   
    ], dtype=np.float32)
    
    # 이미지 좌표 (2D)
    image_points = corners[0].astype(np.float32)
    
    # PnP 문제 해결
    success, rvec, tvec = cv2.solvePnP(
        object_points, image_points, camera_matrix, dist_coeffs
    )
    
    if success:
        return rvec, tvec
    else:
        return None, None


def live_aruco_detection(calibration_data):
    """
    실시간으로 비디오를 받아 ArUco 마커를 검출하고 3D 포즈를 추정하는 함수

    Args:
        calibration_data: 카메라 캘리브레이션 데이터를 포함한 딕셔너리
            - camera_matrix: 카메라 내부 파라미터 행렬
            - dist_coeffs: 왜곡 계수
    """
    # 캘리브레이션 데이터 추출
    camera_matrix = calibration_data['camera_matrix']
    dist_coeffs = calibration_data['dist_coeffs']

    # ArUco 검출기 설정
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_100)
    aruco_params = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(aruco_dict, aruco_params)

    # 마커 크기 설정 (미터 단위)
    marker_size = 0.05  # 예: 5cm = 0.05m

    # 카메라 설정
    cap = cv2.VideoCapture(0)

    # 카메라 초기화 대기
    time.sleep(2)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break

        # 이미지 왜곡 보정
        frame_undistorted = cv2.undistort(frame, camera_matrix, dist_coeffs)

        # 마커 검출
        corners, ids, rejected = detector.detectMarkers(frame_undistorted)

        # 마커가 검출되면 표시 및 포즈 추정
        if ids is not None:
            # 검출된 마커 표시
            cv2.aruco.drawDetectedMarkers(frame_undistorted, corners, ids)

            # 각 마커에 대해 처리
            for i in range(len(ids)):
                # 포즈 추정 (새로운 방법으로 대체)
                rvec, tvec = estimate_pose_single_marker(
                    [corners[i]], marker_size, camera_matrix, dist_coeffs
                )
                
                if rvec is not None and tvec is not None:
                    # 좌표축 표시
                    cv2.drawFrameAxes(frame_undistorted, camera_matrix, dist_coeffs,
                                      rvec, tvec, marker_size/2)

                    # 마커의 3D 위치 표시
                    pos_x = tvec[0][0]
                    pos_y = tvec[1][0]
                    pos_z = tvec[2][0]

                    # 회전 벡터를 오일러 각도로 변환
                    rot_matrix, _ = cv2.Rodrigues(rvec)
                    euler_angles = cv2.RQDecomp3x3(rot_matrix)[0]

                    # 마커 정보 표시
                    corner = corners[i][0]
                    center_x = int(np.mean(corner[:, 0]))
                    center_y = int(np.mean(corner[:, 1]))

                    cv2.putText(frame_undistorted,
                                f"ID: {ids[i][0]}",
                                (center_x, center_y - 40),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.5, (255, 0, 255), 2)

                    cv2.putText(frame_undistorted,
                                f"Pos: ({pos_x:.2f}, {pos_y:.2f}, {pos_z:.2f})m",
                                (center_x, center_y),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.5, (255, 0, 255), 2)

                    cv2.putText(frame_undistorted,
                                f"Rot: ({euler_angles[0]:.1f}, {euler_angles[1]:.1f}, {euler_angles[2]:.1f})deg",
                                (center_x, center_y + 20),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.5, (255, 0, 255), 2)

                    #----------------------------------------------------------- 조건문
                    # 마커와 카메라 간의 거리가 30cm 이하인 경우 "STOP!" 메시지 표시
                    if pos_z < 0.30:  # 30cm 이하일 때
                        # 텍스트 배경 그리기 (배경 색상은 반투명 빨간색)
                        text = "STOP!"
                        text_size, _ = cv2.getTextSize(
                            text, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)
                        text_width, text_height = text_size

                        # 배경 사각형 그리기
                        background_x1 = center_x - text_width // 2 - 10
                        background_y1 = center_y + 40 - text_height // 2 - 10
                        background_x2 = center_x + text_width // 2 + 10
                        background_y2 = center_y + 40 + text_height // 2 + 10
                        cv2.rectangle(frame_undistorted, (background_x1, background_y1),
                                      (background_x2, background_y2), (0, 0, 255), -1)  # 빨간색 배경

                        # 텍스트 그리기
                        cv2.putText(frame_undistorted,
                                    text,
                                    (center_x - text_width // 2,
                                     center_y + 40 + text_height // 2),
                                    cv2.FONT_HERSHEY_SIMPLEX,
                                    1, (255, 255, 255), 2, cv2.LINE_AA)  # 흰색 텍스트
                    
                    else:
                        text = "GO!"
                        text_size, _ = cv2.getTextSize(
                            text, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)
                        text_width, text_height = text_size

                        # 배경 사각형 그리기
                        background_x1 = center_x - text_width // 2 - 10
                        background_y1 = center_y + 40 - text_height // 2 - 10
                        background_x2 = center_x + text_width // 2 + 10
                        background_y2 = center_y + 40 + text_height // 2 + 10
                        cv2.rectangle(frame_undistorted, (background_x1, background_y1),
                                      (background_x2, background_y2), (0, 255, 0), -1)  # 초록색 배경

                        # 텍스트 그리기
                        cv2.putText(frame_undistorted,
                                    text,
                                    (center_x - text_width // 2,
                                     center_y + 40 + text_height // 2),
                                    cv2.FONT_HERSHEY_SIMPLEX,
                                    1, (255, 255, 255), 2, cv2.LINE_AA)  # 흰색 텍스트
                    #-----------------------------------------------------------
                    
                    # 코너 포인트 표시
                    for point in corner:
                        x, y = int(point[0]), int(point[1])
                        cv2.circle(frame_undistorted, (x, y), 4, (0, 0, 255), -1)

        # 프레임 표시
        cv2.imshow('ArUco Marker Detection', frame_undistorted)

        # 'q' 키를 누르면 종료
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # 리소스 해제
    cap.release()
    cv2.destroyAllWindows()


def main():
    # 캘리브레이션 데이터 로드
    try:
        with open('camera_calibration.pkl', 'rb') as f:
            calibration_data = pickle.load(f)
        print("Calibration data loaded successfully")
    except FileNotFoundError:
        print("Error: Camera calibration file not found")
        return
    except Exception as e:
        print(f"Error loading calibration data: {e}")
        return

    print("Starting ArUco marker detection...")
    live_aruco_detection(calibration_data)


if __name__ == "__main__":
    main()
```

</div>
</details>
