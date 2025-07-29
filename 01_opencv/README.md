# openCV 라이브러리에서 이미지 제어

## 목차
1. openCV 라이브러리 설치
2. 이미지 변환

## 1. openCV 라이브러리 설치

1. **가상환경 활성화 후**
   ```bash
   pip list
   ```
   - 설치된 라이브러리 목록 확인
   - 필요시 ```python -m pip install --upgrade pip``` 입력 후 pip 버전 업그레이드

2. **openCV, Numpy, Matplotlib 라이브러리 설치**
   ```bash
   pip install opencv-python
   pip install numpy
   pip install matplotlib
   ```

3. **라이브러리 설치되었는지 확인**
   ```bash
   pip list
   ```
   예시:
   ```bash
   Package         Version
   --------------- -----------
   contourpy       1.3.2
   cycler          0.12.1
   fonttools       4.59.0
   joblib          1.5.1
   kiwisolver      1.4.8
   matplotlib      3.10.3
   numpy           2.2.6
   opencv-python   4.12.0.88
   packaging       25.0
   pandas          2.3.1
   pillow          11.3.0
   pip             25.1.1
   pyparsing       3.2.3
   python-dateutil 2.9.0.post0
   pytz            2025.2
   scikit-learn    1.7.0
   scipy           1.15.3
   setuptools      63.2.0
   six             1.17.0
   threadpoolctl   3.6.0
   tzdata          2025.2
   (myvenv)
   ```

4. **디렉토리 만들기**
   ```bash
   mkdir src    # 소그코드 저장 디렉토리
   mkdir img    # 이미지 저장 디렉토리
   ```
   - 파일 분류를 위해 디렉토리를 구분

## 2. 이미지 제어

1. **이미지 다운로드**
   ```bash
   cd img
   curl -o like_lenna.png https://raw.githubusercontent.com/Cobslab/imageBible/main/image/like_lenna224.png
   ```
   - img 디렉토리에 `like_lenna.png' 이미지 파일 저장

2. **이미지 읽어오기**
   ```bash
   import cv2   # openCV 임포트
   
   # @이미지 불러오기
   img = cv2.imread('../img/like_lenna.png')

   # @이미지를 보여주는 명령어
   cv2.imshow('img', img)

   cv2.waitKey(0)            # 이미지 창을 게속 열어두고 키보드 입력을 기다림
   cv2.destroyAllWindows()   # 열려있는 이미지 창 닫음
   ```
   <img width="224" height="224" alt="image" src="https://github.com/user-attachments/assets/f20f47a8-f55f-4ba2-9f01-ec4a6c9803f1" />
