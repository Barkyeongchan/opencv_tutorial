# [학습 목표 : OpenCV 머신러닝을 활용해 얼굴을 구별하고 표정을 구별 할 수 있다.]

# 하르 캐스케이드 분류기 (Haarcascade) / LBPH 알고리즘

## 목차

1. 하르 캐스케이드 분류기
   - 하르 케스케이드 분류기란?
   - 하르 캐스케이드 얼굴 검출 예시
  
2. LBPH 알고리즘 (Local Binary Patterns Histograms)

## 1. 하르 캐스케이드 분류기 (Haarcascade)

<details>
<summary></summary>
<div markdown="1">

## **1-1. 하르 캐스케이드 분류기란?**

개발자가 **직접 머신러닝 학습 알고리즘을 사용하지 않고도 객체를 검출**할 수 있도록 OpenCV가 제공하는 대표적인 상위 레벨 AP

openCV에서는 `[하르 케스케이드 xml](https://github.com/opencv/opencv/tree/master/data/haarcascades)` 형태로 제공한다.

cv2.CascadeClassifier([filename]) 와 classifier.detectMultiScale(img, scaleFactor, minNeighbors , flags, minSize, maxSize) 함수를 사용한다.

```
classifier = cv2.CascadeClassifier([filename]): 케스케이드 분류기 생성자
```

`filename` : 검출기 저장 파일 경로
`classifier` : 캐스케이드 분류기 객체

```
rect = classifier.detectMultiScale(img, scaleFactor, minNeighbors , flags, minSize, maxSize)
```

`img` : 입력 이미지
`scaleFactor` : 이미지 확대 크기에 제한. 1.3~1.5 (큰값: 인식 기회 증가, 속도 감소)
`minNeighbors` : 요구되는 이웃 수(큰 값: 품질 증가, 검출 개수 감소)
`flags` : 지금 사용안함
`minSize, maxSize` : 해당 사이즈 영역을 넘으면 검출 무시
`rect` : 검출된 영역 좌표 (x, y, w, h)

## **1-2. 하르 캐스케이드 얼굴 검출 예시**

<img width="1134" height="756" alt="image" src="https://github.com/user-attachments/assets/f4e2aba9-a50e-4f53-a7be-5cc7bf5dc263" />

<img width="755" height="500" alt="image" src="https://github.com/user-attachments/assets/51bcbe0a-d8c3-43a1-8621-48722b059d60" />

## **1-3. 캐스케이드 분류기로 얼굴과 눈 검출 실습**

**[1. 코드 생성]**

```python3
import numpy as np
import cv2

# 얼굴 검출을 위한 케스케이드 분류기 생성
face_cascade = cv2.CascadeClassifier('./data/haarcascade_frontalface_default.xml')

# 눈 검출을 위한 케스케이드 분류기 생성
eye_cascade = cv2.CascadeClassifier('./data/haarcascade_eye.xml')

# 검출할 이미지 읽고 그레이 스케일로 변환
img = cv2.imread('../img/children.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# 얼굴 검출
faces = face_cascade.detectMultiScale(gray)

# 검출된 얼굴 순회
for (x,y,w,h) in faces:
    # 검출된 얼굴에 사각형 표시
    cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
    # 얼굴 영역을 ROI로 설정
    roi = gray[y:y+h, x:x+w]
    # ROI에서 눈 검출
    eyes = eye_cascade.detectMultiScale(roi)
    # 검출된 눈에 사각형 표
    for (ex,ey,ew,eh) in eyes:
        cv2.rectangle(img[y:y+h, x:x+w],(ex,ey),(ex+ew,ey+eh),(0,255,0),2)

# 결과 출력 
cv2.imshow('img',img)
cv2.waitKey(0)
cv2.destroyAllWindows()
```

<br><br>

**[2. haarcascade_frontalface_default.xml / haarcascade_eye.xml 다운로드]**

[haarcascade_frontalface_default.xml](https://github.com/opencv/opencv/blob/master/data/haarcascades/haarcascade_frontalface_default.xml)

[haarcascade_eye.xml](https://github.com/opencv/opencv/blob/master/data/haarcascades/haarcascade_eye.xml)

<br><br>

**[3. 코드 실행 결과]**

<img width="510" height="525" alt="image" src="https://github.com/user-attachments/assets/fa58dd80-2660-4e25-9c71-4cd3a0c44f46" />

## **1-4. 카메라 캡쳐로 얼굴과 눈 검출**

**[1. 코드 생성]**

```python3
import cv2

# 얼굴과  검출을 위한 케스케이드 분류기 생성 
face_cascade = cv2.CascadeClassifier('../data/haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('../data/haarcascade_eye.xml')

# 카메라 캡쳐 활성화
cap = cv2.VideoCapture(0)
while cap.isOpened():    
    ret, img = cap.read()  # 프레임 읽기
    if ret:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # 얼굴 검출    
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, \
                                        minNeighbors=5, minSize=(80,80))
        for(x,y,w,h) in faces:
            cv2.rectangle(img, (x,y), (x+w, y+h), (0, 255,0),2)
            roi = gray[y:y+h, x:x+w]
            # 눈 검출
            eyes = eye_cascade.detectMultiScale(roi)
            for i, (ex, ey, ew, eh) in enumerate(eyes):
                if i >= 2:
                    break
                cv2.rectangle(img[y:y+h, x:x+w], (ex,ey), (ex+ew, ey+eh), \
                                    (255,0,0),2)
        cv2.imshow('face detect', img)
    else:
        break
    if cv2.waitKey(5) == 27:
        break
cv2.destroyAllWindows()
```

<br><br>

**[2. 코드 실행 결과]**

<img width="637" height="505" alt="image" src="https://github.com/user-attachments/assets/dbc40f64-a587-4d4c-829f-c4f5972692b4" />

</div>
</details>

## 2. LBPH 알고리즘 (Local Binary Patterns Histograms)

<details>
<summary></summary>
<div markdown="1">

## **2-1. LBPH 알고리즘이란?**

**이미지나 영상에서 검출된 얼굴을 각각 누구인지 인식 할 때 자주 사용되는 알고리즘**

## **2-2. 작동 방식**

**[1. 파라미터 설정]**

_아래의 3가지 파라미터를 먼저 설정해야 함_

`Neighbors(이웃 픽셀 수)` : LBP를 만들 때 사용할 이웃 픽셀 수를 뜻합니다. 이웃 픽셀 수가 많을수록 계산 비용이 높아집니다. 보통 이 값은 8로 설정합니다.

`Grid X(수평 방향 분할 수)` : 수평 방향으로 셀을 분할할 개수를 말합니다. 보통 8로 설정합니다.

`Grid Y(수직 방향 분할 수)` : 수직 방향으로 셀을 분할할 개수를 말합니다. 보통 8로 설정합니다.

<br><br>

**[2. 데이터 준비]**

인식 하려는 사람의 얼굴로 이루어진 데이터를 활용하여 고유한 ID를 생성하여 훈련시킴

<br><br>

**[3. LBP 작업 수행]**

<img width="667" height="186" alt="image" src="https://github.com/user-attachments/assets/5fe776e0-f491-46a6-b2e5-9e14878d1bfb" />

<img width="230" height="144" alt="image" src="https://github.com/user-attachments/assets/d689472c-ba95-47f5-b01a-998a71cebbf2" />

<br><br>

**[4. 히스토그램 만들기]**

<img width="705" height="189" alt="image" src="https://github.com/user-attachments/assets/246bc4da-da3d-404e-8aa9-dbca4d897ad2" />

## **2-3. LBPH 얼굴 인식 실습**

**[1. lbp 샘플 생성 코드]**

```python3
import cv2
import numpy as np
import os 

# 변수 설정 ---①
base_dir = './faces/'   # 사진 저장할 디렉토리 경로
target_cnt = 400        # 수집할 사진 갯수
cnt = 0                 # 사진 촬영 수

# 얼굴 검출 분류기 생성 --- ②
face_classifier = cv2.CascadeClassifier(\
                    './data/haarcascade_frontalface_default.xml')

# 사용자 이름과 번호를 입력 받아 디렉토리 생성 ---③
name = input("Insert User Name(Only Alphabet):")
id = input("Insert User Id(Non-Duplicate number):")
dir = os.path.join(base_dir, name+'_'+ id)
if not os.path.exists(dir):
    os.mkdir(dir)

# 카메라 캡쳐 
cap = cv2.VideoCapture(0)
while cap.isOpened():
    ret, frame = cap.read()
    if ret:
        img = frame.copy()
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        # 얼굴 검출 --- ④
        faces = face_classifier.detectMultiScale(gray, 1.3, 5)
        if len(faces) == 1:
            (x,y,w,h) = faces[0]
            # 얼굴 영역 표시 및 파일 저장 ---⑤
            cv2.rectangle(frame, (x,y), (x+w, y+h), (0,255,0), 1)
            face = gray[y:y+h, x:x+w]
            face = cv2.resize(face, (200, 200))
            file_name_path = os.path.join(dir,  str(cnt) + '.jpg')
            cv2.imwrite(file_name_path, face)
            cv2.putText(frame, str(cnt), (x, y), cv2.FONT_HERSHEY_COMPLEX, \
                             1, (0,255,0), 2)
            cnt+=1
        else:
            # 얼굴 검출이 없거나 1이상 인 경우 오류 표시 ---⑥
            if len(faces) == 0 :
                msg = "no face."
            elif len(faces) > 1:
                msg = "too many face."
            cv2.putText(frame, msg, (10, 50), cv2.FONT_HERSHEY_DUPLEX, \
                            1, (0,0,255))
        cv2.imshow('face record', frame)
        if cv2.waitKey(1) == 27 or cnt == target_cnt: 
            break
cap.release()
cv2.destroyAllWindows()      
print("Collecting Samples Completed.")
```

<br><br>

**[2. 얼굴 검출 결과 확인]**

<img width="279" height="38" alt="image" src="https://github.com/user-attachments/assets/db790b64-ee03-4931-9b73-6ae93daa9eb8" />

<img width="676" height="562" alt="image" src="https://github.com/user-attachments/assets/d26b0d02-b2e9-4c9b-a317-a57b25f93ac1" />

<br><br>

**[3. lbp 얼굴 인식 훈련 코드]**

```python3
import cv2
import numpy as np
import os, glob

# 변수 설정
base_dir = '../faces'
train_data, train_labels = [], []


dirs = [d for d in glob.glob(base_dir+"/*") if os.path.isdir(d)]
print('Collecting train data set:')
for dir in dirs:
    # name_id 형식에서 id를 분리
    id = dir.split('_')[1]          
    files = glob.glob(dir+'/*.jpg')
    print('\t path:%s, %dfiles'%(dir, len(files)))
    for file in files:
        img = cv2.imread(file, cv2.IMREAD_GRAYSCALE)
        # 이미지는 train_data, 아이디는 train_lables에 저장
        train_data.append(np.asarray(img, dtype=np.uint8))
        train_labels.append(int(id))

# NumPy 배열로 변환
train_data = np.asarray(train_data)
train_labels = np.int32(train_labels)

# LBP 얼굴인식기 생성 및 훈련
print('Starting LBP Model training...')
model = cv2.face.LBPHFaceRecognizer_create()
model.train(train_data, train_labels)
model.write('../faces/all_face.xml')
print("Model trained successfully!")
```

<br><br>

**[4. 얼굴 인식 훈련 결과]**

<img width="285" height="71" alt="image" src="https://github.com/user-attachments/assets/153b4709-d7ea-45a5-9ece-0fcf2d032566" />

_xml파일 생성_

<img width="154" height="66" alt="image" src="https://github.com/user-attachments/assets/4bb128a7-9d5b-482f-ab20-337567d3e006" />

</div>
</details>
