# [학습 목표 : OpenCV 머신러닝을 활용해 얼굴을 구별하고 표정을 구별 할 수 있다.]

# 하르 캐스케이드 분류기 (Haarcascade) / LBPH 알고리즘 / DLIB 라이브러리

## 목차

1. 하르 캐스케이드 분류기
   - 하르 케스케이드 분류기란?
   - 하르 캐스케이드 얼굴 검출 예시
   - 캐스케이드 분류기로 얼굴과 눈 검출 실습
   - 카메라 캡쳐로 얼굴과 눈 검출
  
2. LBPH 알고리즘 (Local Binary Patterns Histograms)
   - LBPH 알고리즘이란?
   - 작동 방식
   - LBPH 얼굴 인식 실습
   
3. 개인 프로젝트 (사람 인식 어플리케이션 만들기)
   - 목표
   - 실행 코드
   - 실행 결과

4. DLIB 라이브러리
   - DLIB 라이브러리란?
   - 얼굴 랜드마크 검출 실습
   - 카메라 캡쳐로 얼굴 랜드마크 검출 실습
   - 들로네 삼각형 표시
   - 얼굴 스왑
   - 모자이크 처리

5. 개인 프로젝트 (카메라 캡쳐를 활용한 얼굴 모자이크)
   - 목표
   - 실행 코드
   - 실행 결과
  
6. 졸음감지 프로그램

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

<br><br>

**[5. 훈련된 lbp 얼굴 인식기로 인식 코드]**

```python3
import cv2
import numpy as np
import os, glob

# 변수 설정
base_dir = '../faces'
min_accuracy = 85

# LBP 얼굴 인식기 및 케스케이드 얼굴 검출기 생성 및 훈련 모델 읽기
face_classifier = cv2.CascadeClassifier(\
                '../data/haarcascade_frontalface_default.xml')
model = cv2.face.LBPHFaceRecognizer_create()
model.read(os.path.join(base_dir, 'all_face.xml'))

# 디렉토리 이름으로 사용자 이름과 아이디 매핑 정보 생성
dirs = [d for d in glob.glob(base_dir+"/*") if os.path.isdir(d)]
names = dict([])
for dir in dirs:
    dir = os.path.basename(dir)
    name, id = dir.split('_')
    names[int(id)] = name

# 카메라 캡처 장치 준비 
cap = cv2.VideoCapture(0)
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("no frame")
        break
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    # 얼굴 검출
    faces = face_classifier.detectMultiScale(gray, 1.3, 5)
    for (x,y,w,h) in faces:
        # 얼굴 영역 표시하고 샘플과 같은 크기로 축소
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,255),2)
        face = frame[y:y+h, x:x+w]
        face = cv2.resize(face, (200, 200))
        face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
        # LBP 얼굴 인식기로 예측
        label, confidence = model.predict(face)
        if confidence < 400:
            # 정확도 거리를 퍼센트로 변환
            accuracy = int( 100 * (1 -confidence/400))
            if accuracy >= min_accuracy:
                msg =  '%s(%.0f%%)'%(names[label], accuracy)
            else:
                msg = 'Unknown'
        # 사용자 이름과 정확도 결과 출력
        txt, base = cv2.getTextSize(msg, cv2.FONT_HERSHEY_PLAIN, 1, 3)
        cv2.rectangle(frame, (x,y-base-txt[1]), (x+txt[0], y+txt[1]), \
                    (0,255,255), -1)
        cv2.putText(frame, msg, (x, y), cv2.FONT_HERSHEY_PLAIN, 1, \
                    (200,200,200), 2,cv2.LINE_AA)
    cv2.imshow('Face Recognition', frame)
    
    if cv2.waitKey(1) == 27: #esc 
        break

cap.release()
cv2.destroyAllWindows()
```

<br><br>

**[6. 인식 결과]**

<img width="638" height="508" alt="image" src="https://github.com/user-attachments/assets/28f60f03-ed75-41bb-b580-baf7d446f393" />

</div>
</details>

## 3. 개인 프로젝트 (사람 인식 어플리케이션 만들기)

<details>
<summary></summary>
<div markdown="1">

## **3-1. 목표**

lbp 얼굴 인식을 활용해 사용자마다 지정한 원하는 정보를 불러온다.

## **3-2. 실행 코드**

```python3
import cv2
import os
import json
import numpy as np
import datetime

# --- 경로 설정 ---
BASE_DIR = "../project"
FACES_DIR = os.path.join(BASE_DIR, "faces")
MODELS_DIR = os.path.join(BASE_DIR, "models")
USER_DATA_PATH = os.path.join(FACES_DIR, "user_data.json")
MODEL_PATH = os.path.join(MODELS_DIR, "lbph_model.xml")
LABEL_MAP_PATH = os.path.join(MODELS_DIR, "label_map.json")

# --- 디렉토리 자동 생성 ---
os.makedirs(FACES_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)
if not os.path.exists(USER_DATA_PATH):
    with open(USER_DATA_PATH, "w", encoding="utf-8") as f:
        json.dump({}, f, ensure_ascii=False)

# --- 얼굴 인식기 초기화 ---
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
recognizer = cv2.face.LBPHFaceRecognizer_create()

# --- 사용자 설정 불러오기 ---
with open(USER_DATA_PATH, "r", encoding="utf-8") as f:
    user_data = json.load(f)

def save_user_data():
    with open(USER_DATA_PATH, "w", encoding="utf-8") as f:
        json.dump(user_data, f, indent=4, ensure_ascii=False)

def save_label_map():
    with open(LABEL_MAP_PATH, "w", encoding="utf-8") as f:
        json.dump(label_map, f, indent=4, ensure_ascii=False)

def load_label_map():
    if os.path.exists(LABEL_MAP_PATH):
        with open(LABEL_MAP_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}

# --- 실시간 정보 함수 ---
def get_weather():
    return "☁️ 맑음 28도"

def get_calendar():
    now = datetime.datetime.now()
    return f"📅 오늘은 {now.strftime('%Y년 %m월 %d일')}"

def get_news():
    return "📰 오늘의 뉴스: OpenAI, GPT-5 출시 예정!"

# --- 사용자별 정보 표시(터미널) ---
def print_user_info(user_id):
    info = user_data[user_id]["info"]
    print(f"\n👤 사용자: {user_id}")
    if "날씨" in info:
        print(get_weather())
    if "캘린더" in info:
        print(get_calendar())
    if "뉴스" in info:
        print(get_news())
    print("\n--- [단축키] ---\n[r]: 새 사용자 등록   [u]: 설정 수정   [p]: 정보 재출력   [ESC]: 종료")

def select_user_info():
    options = ["날씨", "캘린더", "뉴스"]
    print("\n✅ 표시할 정보를 선택하세요 (쉼표로 구분):")
    for idx, opt in enumerate(options, 1):
        print(f"{idx}. {opt}")
    choice = input("입력 (예: 1,3): ")
    selected = []
    for idx in choice.split(","):
        try:
            selected.append(options[int(idx.strip()) - 1])
        except:
            pass
    return selected

# --- 새로운 사용자 등록 ---
def register_new_user():
    while True:
        new_id = input("\n🆕 새로운 사용자 ID 입력 (중복 불가): ")
        if new_id in user_data:
            print("⚠️ 이미 존재하는 ID입니다. 다른 ID를 입력하세요.")
        else:
            break

    save_path = os.path.join(FACES_DIR, new_id)
    os.makedirs(save_path, exist_ok=True)

    print("😄 얼굴 데이터를 수집합니다. 정면을 바라보세요...")
    cap = cv2.VideoCapture(0)
    count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x,y,w,h) in faces:
            roi = gray[y:y+h, x:x+w]
            cv2.imwrite(os.path.join(save_path, f"{count}.png"), roi)
            count += 1
            cv2.rectangle(frame, (x,y), (x+w,y+h), (255,0,0), 2)
            cv2.putText(frame, f"Count: {count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

        cv2.imshow("Registering User Face", frame)
        if cv2.waitKey(1) == 27 or count >= 100:
            break

    cap.release()
    cv2.destroyAllWindows()

    # 사용자 정보 설정 입력
    user_data[new_id] = {"info": select_user_info()}
    save_user_data()

    # 모델 학습
    train_model()
    print(f"✅ 사용자 {new_id} 등록 및 학습 완료")

# --- 모델 학습 ---
def train_model():
    faces = []
    labels = []
    for user_id in os.listdir(FACES_DIR):
        user_folder = os.path.join(FACES_DIR, user_id)
        if not os.path.isdir(user_folder):
            continue
        for file in os.listdir(user_folder):
            img_path = os.path.join(user_folder, file)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is not None:
                faces.append(img)
                labels.append(user_id)

    if not faces:
        print("⚠️ 얼굴 이미지가 없습니다. 사용자 등록 후 다시 시도하세요.")
        return False

    global label_map, reverse_label_map
    label_map = {uid: idx for idx, uid in enumerate(set(labels))}
    reverse_label_map = {v:k for k,v in label_map.items()}

    numeric_labels = np.array([label_map[uid] for uid in labels])

    recognizer.train(faces, numeric_labels)
    recognizer.write(MODEL_PATH)
    save_label_map()  # 저장 추가
    print("✅ 모델 학습 완료")
    return True

# --- 사용자 인식용 라벨 매핑 ---
def get_user_from_label(label):
    if label in reverse_label_map:
        return reverse_label_map[label]
    return None

# --- 실행 시작 ---
def main():
    global label_map, reverse_label_map
    label_map = load_label_map()
    reverse_label_map = {v:k for k,v in label_map.items()}

    if os.path.exists(MODEL_PATH):
        recognizer.read(MODEL_PATH)
        print("✅ 학습된 모델 불러오기 성공")
        train_success = True
    else:
        print("⚠️ 학습된 모델이 없습니다. 사용자 등록을 시작합니다.")
        train_success = False

    cap = cv2.VideoCapture(0)
    current_user = None
    printed_users = set()

    if not train_success:
        register_new_user()
        recognizer.read(MODEL_PATH)

    print("\n[스마트미러 시스템 시작]")
    print("[단축키] r: 새 사용자 등록 | u: 설정 수정 | p: 정보 재출력 | ESC: 종료\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("⚠️ 카메라를 읽을 수 없습니다.")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x,y,w,h) in faces:
            roi = gray[y:y+h, x:x+w]
            try:
                label, confidence = recognizer.predict(roi)
                user_id = get_user_from_label(label)
                if user_id:
                    cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), 2)
                    cv2.putText(frame, f"{user_id}", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

                    if user_id != current_user:
                        current_user = user_id
                        if user_id not in printed_users:
                            print_user_info(user_id)
                            printed_users.add(user_id)
            except:
                pass

        cv2.imshow("Smart Mirror", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == 27:
            break
        elif key == ord('r'):
            register_new_user()
            recognizer.read(MODEL_PATH)
            printed_users.clear()
            current_user = None
        elif key == ord('u') and current_user:
            print(f"\n⚙️ [{current_user}] 설정 변경:")
            user_data[current_user]["info"] = select_user_info()
            save_user_data()
            print_user_info(current_user)
        elif key == ord('p') and current_user:
            print_user_info(current_user)

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
```

**[1. 작동 순서]**

```
1. 프로그램 실행
2. 학습된 모델과 사용자 데이터 로드
3. 실시간 얼굴 인식 진행
4. 새로운 사용자가 인식되면 터미널에서 정보 입력
5. 인식된 사용자 이름 및 정보 우측 OpenCV 창에 표시
6. 단축키 `u`로 기존 사용자 정보 갱신 가능
7. 단축키 `r`로 새로운 사용자 등록 가능
```

<br><br>

**[2. user_data.json 사용자 정보 저장 구조]**

```json
{
  "1": {
    "name": "karina",
    "weather": "날씨"
  },
  "2": {
    "name": "park",
    "calendar": "날짜",
    "news": "뉴스"
  }
}
```

<br><br>

**[3. 사용자 등록]**

- **ID(이름) 중복 방지**: 이미 등록된 이름으로는 추가 등록 불가합니다.
- **얼굴 100장 촬영 후 저장**: LBPH 학습을 위한 데이터 수집.
- **사용자 정보 저장**: 사용자가 선택한 날씨, 뉴스, 캘린더 정보는 `user_data.json`에 저장됩니다.

```python
def register_new_user():
    ...
```

<br><br>

**[4. 얼굴 인식]**

- **실시간 얼굴 탐지 및 예측**: 웹캠을 통해 얼굴을 감지하고 등록된 모델과 비교합니다.
- **사용자 이름 표시**: 인식된 사용자의 이름이 OpenCV 화면 오른쪽에 표시됩니다.
- **중복 출력 방지**: 동일 사용자는 이미 출력된 경우 터미널에 중복 출력하지 않습니다.
- **특정 단축키로 출력 갱신 가능**: 이전 사용자도 키보드 입력으로 정보 다시 출력 가능.

```python
label, confidence = recognizer.predict(roi)
user_id = get_user_from_label(label)
```

<br><br>

**[5. 사용자별 정보 출력]**

- `get_weather()`, `get_calendar()`, `get_news()` 함수로 사용자별 텍스트 정보 생성
- **선택된 항목만 출력**: 각 사용자가 사전에 선택한 항목만 출력됩니다.

```python
def print_user_info(user_id):
    ...
```

<br><br>

**[6. 데이터 저장]**

- **사용자 정보 저장**: `user_data.json`에 사용자별 출력 항목 저장
- **라벨 매핑 저장**: `label_map.json`에 ID-이름 대응 정보 저장
- **얼굴 인식 모델 저장**: 학습된 LBPH 모델은 `lbph_model.xml`로 저장됩니다.
- **재실행 시에도 인식 가능**: 저장된 모델과 데이터로 실행 후에도 바로 사용자 인식 가능

```python
def save_user_data():
    ...
```

## **3-3. 실행 결과**

**[1. 최초 실행 시]**

<img width="364" height="22" alt="image" src="https://github.com/user-attachments/assets/95fee64b-d9d5-456d-8b53-4329332c9f02" />

<br><br>

**[2. 사용자 ID 정의]**

<img width="495" height="97" alt="image" src="https://github.com/user-attachments/assets/0e091d51-da5a-4f86-9dfc-5e8407fe8ebe" />

<br><br>

**[3. 얼굴 검출 및 학습]**

<img width="635" height="507" alt="image" src="https://github.com/user-attachments/assets/10b7bb01-4155-47f1-a862-f0553816330b" />

<br><br>

**[4. 각 ID마다 표시할 정보 선택]**

<img width="303" height="97" alt="image" src="https://github.com/user-attachments/assets/58e6cbf9-0e05-4f57-8301-9cae7cf02ad1" />

<br><br>

**[5. 각 ID에 맞춰 정보 출력]**

<img width="640" height="607" alt="image" src="https://github.com/user-attachments/assets/b0ef2bd0-72db-423b-b2f4-158288bbd0ea" />

<br><br>

**[6. 새로운 ID 및 ID에 맞춘 정보 출력]**

<img width="640" height="577" alt="image" src="https://github.com/user-attachments/assets/dec43836-2f3e-42f0-93d8-a1afb76414b5" />

</div>
</details>

## 4. Dlib 라이브러리

<details>
<summary></summary>
<div markdown="1">

## **4-1. DLIB 라이브러리란?**

`dlib`은 **얼굴 인식 및 머신 러닝 기능을 포함**한 C++ 기반의 고성능 라이브러리이다.

```terminal
pip instll dlib-bin
```

**[API]**

`detector` : dlib.get_frontal_face_detector(): 얼굴 검출기 생성

`predictor` : dlib.shap_predictor(file): 랜드마크 검출기 생성

`rects` : detector(img) : 얼굴 검출

`shape` : predictor(img, rect) : 랜드마크 검출

<img width="1856" height="1496" alt="image" src="https://github.com/user-attachments/assets/68203192-0580-4b38-b8f8-a9bbfb1baa88" />

## **4-2. 얼굴 랜드마크 검출 실습**

```python3
import cv2
import dlib

# 얼굴 검출기와 랜드마크 검출기 생성
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('./shape_predictor_68_face_landmarks.dat')

img = cv2.imread("../img/like_lenna.png")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# 얼굴 영역 검출
faces = detector(gray)
for rect in faces:
    # 얼굴 영역을 좌표로 변환 후 사각형 표시
    x,y = rect.left(), rect.top()
    w,h = rect.right()-x, rect.bottom()-y
    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 1)

    # 얼굴 랜드마크 검출
    shape = predictor(gray, rect)
    for i in range(68):
        # 부위별 좌표 추출 및 표시
        part = shape.part(i)
        cv2.circle(img, (part.x, part.y), 2, (0, 0, 255), -1)
        cv2.putText(img, str(i), (part.x, part.y), cv2.FONT_HERSHEY_PLAIN, \
                                         0.5,(255,255,255), 1, cv2.LINE_AA)

cv2.imshow("face landmark", img)
cv2.waitKey(0)
```

<img width="510" height="525" alt="image" src="https://github.com/user-attachments/assets/afbc7225-1526-4e8b-b1bf-bab76258cf07" />

<img width="639" height="502" alt="image" src="https://github.com/user-attachments/assets/adbc5c82-c4db-481e-8968-7ec3b0f897dd" />


## **4-3. 카메라 캡쳐로 얼굴 랜드마크 검출 실습**

```python3
import cv2
import dlib

# 얼굴 검출기와 랜드마크 검출기 생성
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('./shape_predictor_68_face_landmarks.dat')

cap = cv2.VideoCapture(0)
#cap.set(cv2.cv2.CAP_PROP_FRAME_WIDTH, 480)
#cap.set(cv2.cv2.CAP_PROP_FRAME_HEIGHT, 320)

while cap.isOpened():
    ret, img = cap.read()
    if not ret:
        print('no frame.');break
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # 얼굴 영역 검출
    faces = detector(gray)
    for rect in faces:

        # 얼굴 영역을 좌표로 변환 후 사각형 표시
        x,y = rect.left(), rect.top()
        w,h = rect.right()-x, rect.bottom()-y
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 1)
    
        # 얼굴 랜드마크 검출
        shape = predictor(gray, rect)
        for i in range(68):
            # 부위별 좌표 추출 및 표시
            part = shape.part(i)
            cv2.circle(img, (part.x, part.y), 2, (0, 0, 255), -1)
#            cv2.putText(img, str(i), (part.x, part.y), cv2.FONT_HERSHEY_PLAIN, 0.5,(255,255,255), 1, cv2.LINE_AA)
    
    cv2.imshow("face landmark", img)
    if cv2.waitKey(1)== 27:
        break
cap.release()
```

<img width="638" height="511" alt="image" src="https://github.com/user-attachments/assets/f42a11c0-99b5-442b-8254-4f81519630ed" />

## **4-4. 들로네 삼각형 표시**

```pyhon3
import cv2
import numpy as np
import dlib

# 얼굴 검출기와 랜드마크 검출기 생성
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('./shape_predictor_68_face_landmarks.dat')

img = cv2.imread("../img/man_face.jpg")
h, w = img.shape[:2]
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# 얼굴 영역 검출
rects = faces = detector(gray)

points = []
for rect in rects:
    # 랜드마크 검출
    shape = predictor(gray, rect)
    for i in range(68):
        part = shape.part(i)
        points.append((part.x, part.y))
        

# 들로네 삼각 분할 객체 생성
x,y,w,h = cv2.boundingRect(np.float32(points))
subdiv = cv2.Subdiv2D((x,y,x+w,y+h))
# 랜드마크 좌표 추가
subdiv.insert(points)
# 들로네 삼각형 좌표 계산
triangleList = subdiv.getTriangleList()
# 들로네 삼각형 그리기
h, w = img.shape[:2]
cnt = 0
for t in triangleList :
    pts = t.reshape(-1,2).astype(np.int32)
    # 좌표 중에 이미지 영역을 벗어나는 것을 제외(음수 등)
    if (pts < 0).sum() or (pts[:, 0] > w).sum() or (pts[:, 1] > h).sum():
        print(pts) 
        continue
    cv2.polylines(img, [pts], True, (255, 255,255), 1, cv2.LINE_AA)
    cnt+=1
print(cnt)


cv2.imshow("Delaunay",img)
cv2.waitKey(0)
```

<img width="639" height="505" alt="image" src="https://github.com/user-attachments/assets/478a42f3-3c98-441e-854c-b5e637c4810e" />

## **4-5. 얼굴 스왑**

```python3
import cv2
import numpy as np
import dlib
import sys

# 얼굴 검출기와 랜드마크 검출기 생성
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('./shape_predictor_68_face_landmarks.dat')

# 얼굴 및 랜드마크 검출해서 좌표 반환하는 함수
def getPoints(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    rects = detector(gray)
    points = []
    for rect in rects:
        shape = predictor(gray, rect)
        for i in range(68):
            part = shape.part(i)
            points.append((part.x, part.y))
    return points    

# 랜드마크 좌표로 들로네 삼각형 반환
def getTriangles(img, points):
    w,h = img2.shape[:2]
    subdiv = cv2.Subdiv2D((0,0,w,h));
    subdiv.insert(points) 
    triangleList = subdiv.getTriangleList();
    triangles = []
    for t in triangleList:        
        pt = t.reshape(-1,2)
        if not (pt < 0).sum() and not (pt[:, 0] > w).sum() \
                              and not (pt[:, 1] > h).sum(): 
            indice = []
            for i in range(0, 3):
                for j in range(0, len(points)):                    
                    if(abs(pt[i][0] - points[j][0]) < 1.0 \
                        and abs(pt[i][1] - points[j][1]) < 1.0):
                        indice.append(j)    
            if len(indice) == 3:                                                
                triangles.append(indice)
    return triangles

# 삼각형 어핀 변환 함수
def warpTriangle(img1, img2, pts1, pts2):
    x1,y1,w1,h1 = cv2.boundingRect(np.float32([pts1]))
    x2,y2,w2,h2 = cv2.boundingRect(np.float32([pts2]))
    
    roi1 = img1[y1:y1+h1, x1:x1+w1]
    roi2 = img2[y2:y2+h2, x2:x2+w2]
    
    offset1 = np.zeros((3,2), dtype=np.float32)
    offset2 = np.zeros((3,2), dtype=np.float32)
    for i in range(3):
        offset1[i][0], offset1[i][1] = pts1[i][0]-x1, pts1[i][1]-y1
        offset2[i][0], offset2[i][1] = pts2[i][0]-x2, pts2[i][1]-y2
    
    mtrx = cv2.getAffineTransform(offset1, offset2)
    warped = cv2.warpAffine( roi1, mtrx, (w2, h2), None, \
                        cv2.INTER_LINEAR, cv2.BORDER_REFLECT_101 )
    
    mask = np.zeros((h2, w2), dtype = np.uint8)
    cv2.fillConvexPoly(mask, np.int32(offset2), (255))
    
    warped_masked = cv2.bitwise_and(warped, warped, mask=mask)
    roi2_masked = cv2.bitwise_and(roi2, roi2, mask=cv2.bitwise_not(mask))
    roi2_masked = roi2_masked + warped_masked
    img2[y2:y2+h2, x2:x2+w2] = roi2_masked

if __name__ == '__main__' :
    # 이미지 읽기
    img1 = cv2.imread('../img/boy_face.jpg')
    img2 = cv2.imread('../img/girl_face.jpg')
    cv2.imshow('img1', img1)
    cv2.imshow('img2', img2)
    img_draw = img2.copy()
    
    # 각 이미지에서 얼굴 랜드마크 좌표 구하기
    points1 = getPoints(img1)
    points2 = getPoints(img2)
    
    # 랜드마크 좌표로 볼록 선체 구하기
    hullIndex = cv2.convexHull(np.array(points2), returnPoints = False)
    hull1 = [points1[int(idx)] for idx in hullIndex]
    hull2 = [points2[int(idx)] for idx in hullIndex]
    
    # 볼록 선체 안 들로네 삼각형 좌표 구하기
    triangles = getTriangles(img2, hull2)
    
    # 각 삼각형 좌표로 삼각형 어핀 변환   
    for i in range(0, len(triangles)):
        t1 = [hull1[triangles[i][j]] for j in range(3)]
        t2 = [hull2[triangles[i][j]] for j in range(3)]
        warpTriangle(img1, img_draw, t1, t2)
   
    # 볼록선체를 마스크로 써서 얼굴 합성
    mask = np.zeros(img2.shape, dtype = img2.dtype)  
    cv2.fillConvexPoly(mask, np.int32(hull2), (255, 255, 255))
    r = cv2.boundingRect(np.float32([hull2]))    
    center = ((r[0]+int(r[2]/2), r[1]+int(r[3]/2)))
    output = cv2.seamlessClone(np.uint8(img_draw), img2, mask, center, \
                                cv2.NORMAL_CLONE)
    
    cv2.imshow("Face Swapped", output)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
```

<img width="935" height="315" alt="image" src="https://github.com/user-attachments/assets/f58791e1-e68c-4756-9e57-e8a0bf5e77ad" />

## **4-6. 모자이크 처리**

```python3
import cv2

rate = 15               # 모자이크에 사용할 축소 비율 (1/rate)
win_title = 'mosaic'    # 창 제목
img = cv2.imread('../img/like_lenna.png')    # 이미지 읽기

while True:
    x,y,w,h = cv2.selectROI(win_title, img, False) # 관심영역 선택
    if w and h:
        roi = img[y:y+h, x:x+w]   # 관심영역 지정
        roi = cv2.resize(roi, (w//rate, h//rate)) # 1/rate 비율로 축소
        # 원래 크기로 확대
        roi = cv2.resize(roi, (w,h), interpolation=cv2.INTER_AREA)  
        img[y:y+h, x:x+w] = roi   # 원본 이미지에 적용
        cv2.imshow(win_title, img)
    else:
        break
cv2.destroyAllWindows()
```

<img width="510" height="525" alt="image" src="https://github.com/user-attachments/assets/06811cad-c590-4b37-a902-bf6e05b9a5ae" />

<img width="510" height="525" alt="image" src="https://github.com/user-attachments/assets/e2c77240-66d3-4519-a773-a14d863d3a3d" />

</div>
</details>

## 5. 개인 프로젝트 (카메라 캡쳐를 활용한 얼굴 모자이크)

<details>
<summary></summary>
<div markdown="1">

## **5-1. 목표**

카메라 캡쳐를 통해 찍은 영상에서 **얼굴을 자동으로 인식하고 모자이크하는**프로그램 생성

## **5-2. 실행 코드**

```python3
import cv2
import dlib

# 얼굴 검출기와 랜드마크 검출기 생성
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('./shape_predictor_68_face_landmarks.dat')

cap = cv2.VideoCapture(0)
rate = 15  # 모자이크 비율

while cap.isOpened():
    ret, img = cap.read()
    if not ret:
        print('no frame.');break
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # 얼굴 영역 검출
    faces = detector(gray)

    for rect in faces:

        # 얼굴 영역을 좌표로 변환 후 사각형 표시
        x,y = rect.left(), rect.top()
        w,h = rect.right()-x, rect.bottom()-y

        roi = img[y:y+h, x:x+w]
        if roi.size == 0:
            continue
        roi = cv2.resize(roi, (w // rate, h // rate))
        roi = cv2.resize(roi, (w, h), interpolation=cv2.INTER_AREA)
        img[y:y+h, x:x+w] = roi
    
    cv2.imshow("mosaic", img)
    if cv2.waitKey(1)== 27:
        break

cv2.destroyAllWindows()
cap.release()
```

## **5-3. 실행 결과**

<img width="640" height="511" alt="image" src="https://github.com/user-attachments/assets/5d9349ed-663f-4f76-9503-7130e6b65b12" />

</div>
</details>

## 6. 졸음감지 프로그램

<details>
<summary></summary>
<div markdown="1">

```python3
import cv2
import dlib
from utils.landmark_utils import get_eye_landmarks, calculate_ear

# 설정값은 settings.py에서 가져온다고 가정
import config.settings as settings

# 상태 변수
consecutive_frames = 0

def detect_faces(frame, detector):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)
    return faces

def get_landmarks(frame, face_rect, predictor):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    shape = predictor(gray, face_rect)
    return shape

def calculate_ear_from_landmarks(landmarks):
    left_eye = get_eye_landmarks(landmarks, settings.LEFT_EYE)
    right_eye = get_eye_landmarks(landmarks, settings.RIGHT_EYE)
    left_ear = calculate_ear(left_eye)
    right_ear = calculate_ear(right_eye)
    return (left_ear + right_ear) / 2.0

def check_drowsiness(ear_value):
    global consecutive_frames
    if ear_value < settings.EAR_THRESHOLD:
        consecutive_frames += 1
    else:
        consecutive_frames = 0

    if consecutive_frames == 0:
        return 'NORMAL'
    elif consecutive_frames < 10:
        return 'DROWSY'
    elif consecutive_frames < 20:
        return 'ALERT'
    else:
        return 'DANGER'

def draw_results(frame, landmarks, ear_value, drowsiness_level):
    left_eye = get_eye_landmarks(landmarks, settings.LEFT_EYE)
    right_eye = get_eye_landmarks(landmarks, settings.RIGHT_EYE)

    for (x, y) in left_eye + right_eye:
        cv2.circle(frame, (x, y), 2, settings.GREEN, -1)

    cv2.putText(frame, f"EAR: {ear_value:.3f}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, settings.GREEN, 2)

    color_map = {
        'NORMAL': settings.GREEN,
        'DROWSY': settings.YELLOW,
        'ALERT': (0, 165, 255),  # 주황색
        'DANGER': settings.RED
    }
    color = color_map.get(drowsiness_level, settings.GREEN)
    cv2.putText(frame, drowsiness_level, (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 3)

def main():
    global consecutive_frames
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(settings.LANDMARK_MODEL_PATH)
    cap = cv2.VideoCapture(0)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        faces = detect_faces(frame, detector)
        if len(faces) > 0:
            landmarks = get_landmarks(frame, faces[0], predictor)
            ear_value = calculate_ear_from_landmarks(landmarks)
            drowsiness_level = check_drowsiness(ear_value)
            draw_results(frame, landmarks, ear_value, drowsiness_level)
        else:
            consecutive_frames = 0
            cv2.putText(frame, "No face detected", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, settings.RED, 2)

        cv2.imshow('Drowsiness Detection', frame)
        if cv2.waitKey(1) == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
```

<img width="637" height="507" alt="image" src="https://github.com/user-attachments/assets/7992cc7d-44f3-46cc-a372-25413424dc74" />

<img width="636" height="507" alt="image" src="https://github.com/user-attachments/assets/05761adf-1181-40f9-b785-44a43677f13a" />

</div>
</details>
