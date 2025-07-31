<<<<<<< HEAD
# opencv 파일 입출력 예제
=======
# openCV 라이브러리에서 이미지 제어

## 목차
1. openCV 라이브러리 설치
2. 이미지 변환
3. 도형 그리기
4. 개인 프로젝트
   

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
   ```python3
   import cv2   # openCV 임포트
   
   # @이미지 불러오기
   img = cv2.imread('../img/like_lenna.png')

   # @이미지를 보여주는 명령어
   cv2.imshow('img', img)

   cv2.waitKey(0)            # 이미지 창을 게속 열어두고 키보드 입력을 기다림
   cv2.destroyAllWindows()   # 열려있는 이미지 창 닫음
   ```
   <img width="224" height="224" alt="image" src="https://github.com/user-attachments/assets/f20f47a8-f55f-4ba2-9f01-ec4a6c9803f1" />

3. **사이즈 변환**
   ```python3
   import cv2   # openCV 임포트
   
   # @이미지 불러오기
   img = cv2.imread('../img/like_lenna.png')
   small_img = cv2.resize(img,(100,100))

   # @이미지를 보여주는 명령어
   cv2.imshow('small_img', small_img)

   cv2.waitKey(0)
   cv2.destroyAllWindows()
   ```
   <img width="100" height="100" alt="image" src="https://github.com/user-attachments/assets/08f34306-10d9-4403-80a6-aadaf0cdee31" />

4. **변화하는 인자를 받아서 리사이징**
   ```python3
   import cv2
   import numpy as np   # 넘파이 임포트

   img = cv2.imread('../img/like_lenna.png')

   # @이미지 크기를 인자로 받음
   new_height = 300
   new_width = 300

   # @빈 배열을 만들고 사이즈 조정
   dst = np.zeros((new_height, new_width), dtype=np.uint8)
   cv2.resize(image, (new_width, new_height), dst=dst)

   cv2_imshow('dst', dst)

   cv2.waitKey(0)
   cv2.destroyAllWindows()
   ```
   <img width="300" height="300" alt="image" src="https://github.com/user-attachments/assets/88f29bf2-59fc-498b-a418-9da30fc0e44c" />

5. **배율로 사이즈 변환**
   ```python3
   import cv2
   
   img = cv2.imread('../img/like_lenna.png')

   # @이미지 사이즈를 배율로 조정
   big_img = cv2.resize(img,dsize=None,fx=2,fy=2,)

   cv2.imshow('big_img', big_img)

   cv2.waitKey(0)
   cv2.destroyAllWindows()
   ```
   <img width="448" height="448" alt="image" src="https://github.com/user-attachments/assets/e179a63f-844f-4692-89d4-d6f59bd427a8" />

6. **대칭 변환**
   ```python3
   import cv2
   
   img = cv2.imread('../img/like_lenna.png')

   # @이미지 대칭 변환
   fliped_img = cv2.flip(img,0)   # 0 = 상하 대칭

   cv2.imshow('fliped_img', fliped_img)

   cv2.waitKey(0)
   cv2.destroyAllWindows() 
   ```
   <img width="224" height="224" alt="image" src="https://github.com/user-attachments/assets/7836676b-e9ea-4f7f-8ed7-86c879e0dfdd" />
   
   ```python3
   # @이미지 대칭 변환
   fliped_img = cv2.flip(img,1)   # 1 = 좌우 대칭
   ```
   <img width="224" height="224" alt="image" src="https://github.com/user-attachments/assets/bcb4c900-fd9a-4955-90ac-072e2cfbe019" />

6. **회전 변환**
   ```python3
   import cv2
   import numpy as np
   
   img = cv2.imread('../img/like_lenna.png')

   # @이미지 회전 변환
   height, width = img.shape   # 이미지 크기 가져오기

   matrix = cv2.getRotationMatrix2D((width/2, height/2), 90, 1)   # 회전 변환 행렬 생성
   # (width/2, height/2) = 이미지 중심 좌표, 90 = 회전 각도, 1 = 크기   
   result = cv2.warpAffine(img, matrix, (width, height))        # 회전 변환 적용

   cv2.imshow('result', result)

   cv2.waitKey(0)
   cv2.destroyAllWindows() 
   ```
   <img width="224" height="224" alt="image" src="https://github.com/user-attachments/assets/3a58c653-9be9-49cd-aa86-1c5ef04d42fe" />

6. **이미지 자르기**
   ```python3
   import cv2
   
   img = cv2.imread('../img/like_lenna.png')

   # @이미지 자르기
   cuted_img = cv2_imshow(img[:100,:100])

   cv2.imshow('cuted_img', cuted_img)

   cv2.waitKey(0)
   cv2.destroyAllWindows() 
   ```
   <img width="100" height="100" alt="image" src="https://github.com/user-attachments/assets/092cb808-16fa-4d67-bff5-d6c5c5b33223" />
   
   ```python3
   cuted_img = cv2_imshow(img[50:150,50:150])
   ```
   <img width="100" height="100" alt="image" src="https://github.com/user-attachments/assets/abeef95d-4b04-416c-a239-fbedf6e5726b" />

   ```python3
   cuted_img = cv2_imshow(img[50:150,50:150])
   ```

## 3. 도형 그리기

1. **직선**
   ```python3
   import cv2
   import numpy as np

   # @배경이미지 만들기
   space = np.zeros((500, 1000), dtype=np.uint8)   # 높이 500, 너비 1000 픽셀의 검은색(0) 배경 이미지 생성
   line_color = 255   # 선 색깔 지정

   # @선 그리기
   space = cv2.line(space, (100, 100), (800, 400), line_color, 3, 1)

   '''
   cv2.line()함수 사용
   space : 이미지 위에 선을 그림
   (100, 100) : 시작점 좌표
   (800, 400) : 끝점 좌표
   line_color : 선 색깔 (255)
   3 : 선 두께
   1 : 선 종류 (cv2.LINE_AA)
   '''

   cv2.imshow('line', space)

   cv2.waitKey(0)
   cv2.destroyAllWindows()
   ```
   <img width="1000" height="500" alt="image" src="https://github.com/user-attachments/assets/56171c31-0b0e-46a4-8793-4fbe27a75b0b" />

2. **원**
   ```python3
   import cv2
   import numpy as np

   space = np.zeros((500, 1000), dtype=np.uint8)
   color = 255   # 색상 지정

   # @원 그리기
   space = cv2.circle(space, (600, 200), 100, color, 4, 1)

   '''
   cv2.circle()함수 사용
   space : 이미지 위에 선을 그림
   (600, 200) : 원의 중심 좌표
   100 : 반지름
   color : 색깔 (255)
   4 : 선 두께
   1 : 선 종류 (cv2.LINE_AA)
   '''

   cv2.imshow('circle', space)

   cv2.waitKey(0)
   cv2.destroyAllWindows()
   ```
   <img width="1000" height="500" alt="image" src="https://github.com/user-attachments/assets/67f24b50-9466-4d06-94a8-94d854ea5d38" />

3. **사각형**
   ```python3
   import cv2
   import numpy as np

   space = np.zeros((500, 1000), dtype=np.uint8)
   color = 255   # 색상 지정

   # @사각형 그리기
   space = cv2.rectangle(space, (500, 200), (800, 400), color, 5, 1)

   '''
   cv2.rectangle()함수 사용
   space : 이미지 위에 선을 그림
   (500, 200) : 좌상 좌표 = 왼쪽 상단
   (800, 400) : 우하 좌표 = 오른쪽 하단
   color : 색깔 (255)
   5 : 선 두께
   1 : 선 종류 (cv2.LINE_AA)
   '''

   cv2.imshow('rectangle', space)

   cv2.waitKey(0)
   cv2.destroyAllWindows()
   ```
   <img width="1388" height="768" alt="image" src="https://github.com/user-attachments/assets/b7ffb528-90a7-43ad-96d8-95e9c5f53a36" />

3. **타원**
   ```python3
   import cv2
   import numpy as np

   space = np.zeros((500, 1000), dtype=np.uint8)
   color = 255   # 색상 지정

   # @타원 그리기
   space = cv2.ellipse(space, (500, 300), (300, 200), 0, 90, 250, color, 4)

   '''
   cv2.ellipse()함수 사용
   space : 이미지 위에 선을 그림
   (500, 300) : 타원의 중심 좌표
   (300, 200) : 타원의 반지름 크기
   0 : 타원의 회전 각도
   90 : 타원 그리기 시작 각도
   250 : 타원 그리기 끝 각도
   color : 색깔 (255)
   4 : 선 두께
   '''

   cv2.imshow('ellipse', space)

   cv2.waitKey(0)
   cv2.destroyAllWindows()
   ```
   <img width="1388" height="768" alt="image" src="https://github.com/user-attachments/assets/8b42b171-6daa-4dcf-90fb-16673e1b9f44" />

4. **다각형**
   ```python3
   import cv2
   import numpy as np

   space = np.zeros((500, 1000), dtype=np.uint8)
   color = 255   # 색상 지정

   # @다각형 그리기
   obj1 = np.array([[300, 500], [500, 500], [400, 600], [200, 600]])   # 각 좌표값을 모서리로 가진 다각형 (이 코드에선 사다리꼴)
   obj2 = np.array([[600, 500], [800, 500], [700, 200]])               # 각 좌표값을 모서리로 가진 다각형 (이 코드에선 삼각형)

   # @다각형 정의
   space = cv2.polylines(space, [obj1], True, color, 3)

   '''
   cv2.polylines()함수 사용
   space : 이미지 위에 선을 그림
   [obj1] : 좌표 배열 리스트
   (300, 200) : 타원의 반지름 크기
   True : 마지막 점과 첫 점 연결 유무 (True = 연결 / False = 연결하지 않음)
   color : 색깔 (255)
   3 : 선 두께
   '''

   space = cv2.fillPoly(space, [obj2], color, 1)

   '''
   cv2.fillPoly()함수 사용
   space : 이미지 위에 선을 그림
   [obj2] : 좌표 배열 리스트
   color : 내부 채울 색깔
   1 : 선 타입
   '''

   cv2.imshow('polygon', space)

   cv2.waitKey(0)
   cv2.destroyAllWindows()
   ```
   <img width="1388" height="768" alt="image" src="https://github.com/user-attachments/assets/e7650f4f-16af-49a6-af94-cf9fd06f5ca2" />

5. **격자**
   ```python3
   import cv2
   import numpy as np

   space = np.zeros((500, 1000), dtype=np.uint8)

   # @격자 간격 및 색상 설정
   grid_spacing = 50
   grid_color = 225

   # @격자 그리기
   for x in range(0, space.shape[1], grid_spacing):
      cv2.line(space, (x, 0), (x, space.shape[0]), grid_color, 1)

   for y in range(0, space.shape[0], grid_spacing):
      cv2.line(space, (0, y), (space.shape[1], y), grid_color, 1)

   '''
   space.shape[1] : 이미지 가로 너비 width
   space.shape[0] : 이미지 세로 높이 height
   cv2.line()함수 사용
   '''

   cv2.imshow('grid', space)

   cv2.waitKey(0)
   cv2.destroyAllWindows()
   ```
   <img width="1388" height="768" alt="image" src="https://github.com/user-attachments/assets/0a00eba5-b14a-4701-b35e-615ad3bca274" />


## 4. 개인 프로젝트
목표 : 이미지 제어와 도형 그리기를 활용하여 like_lenna.png 이미지를 꾸며본다.

```python3
import cv2
import numpy as np

image = cv2.imread('../img/like_lenna.png')   # 이미지 불러오기
lenna_img = cv2.resize(image, (500, 500))     # 이미지 크기 조절
color = (147, 20, 255)                        # 색상 정의

'''
# @격자 만들기 - 테두리 선을 만들기 편하게 격자를 사용한 후 안 보이게 주석처리 함
grid_spacing = 25                                                # 격자 간격
for x in range(0, space.shape[1], grid_spacing):                 # 수직선 그리기
    cv2.line(lenna_img, (x, 0), (x, space.shape[0]), color, 1)

for y in range(0, space.shape[0], grid_spacing):                 # 수평선 그리기
    cv2.line(lenna_img, (0, y), (space.shape[1], y), color, 1)
'''

# @테두리선 그리기
line1 = np.array([[360,0], [395,60], [425, 65], [450, 70], [465, 75], [475, 85],
                  [475, 100],[450,140],[425,160],[410,170],[400,190],[405,210],
                  [410,225],[390,270],[395,280],[385,320],[380,340],[375,350],
                  [340,360],[320,390],[315,400],[335,445],[335,475],[350,500],
                  [35,500],[65,450],[120,400],[140,340],[150,340],[170,325],
                  [170,300],[160,230],[125,225],[100,200],[90,175],[100,150],
                  [125,125],[155,100],[180,25],[195,0]])

cv2.polylines(lenna_img, [line1], True, color, 3)   # 윤곽선 정의

cv2.imshow('Image Window', lenna_img)

cv2.waitKey(0)
cv2.destroyAllWindows()
```

<img width="627" height="627" alt="practice_like_lenna" src="https://github.com/user-attachments/assets/045b2d34-595c-42a4-bebd-6f9a5743090a" />

>>>>>>> 8e665caf898e29312c928bdf80d0c93460ccf648
