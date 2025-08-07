import cv2
import os
import json
import numpy as np
import datetime
import requests

# --- 경로 설정 ---
BASE_DIR = "../project"
FACES_DIR = os.path.join(BASE_DIR, "faces")
MODELS_DIR = os.path.join(BASE_DIR, "models")
USER_DATA_PATH = os.path.join(FACES_DIR, "user_data.json")
MODEL_PATH = os.path.join(MODELS_DIR, "lbph_model.xml")

# --- 디렉토리 자동 생성 ---
os.makedirs(FACES_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)
if not os.path.exists(USER_DATA_PATH):
    with open(USER_DATA_PATH, "w") as f:
        json.dump({}, f)

# --- 얼굴 인식기 초기화 ---
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
recognizer = cv2.face.LBPHFaceRecognizer_create()

# --- 사용자 설정 불러오기 ---
with open(USER_DATA_PATH, "r") as f:
    user_data = json.load(f)

def save_user_data():
    with open(USER_DATA_PATH, "w") as f:
        json.dump(user_data, f, indent=4)

# --- 실시간 정보 ---
def get_weather():
    # 예시용 - 실제 API 키 넣을 것
    return "☁️ 맑음 28도"

def get_calendar():
    now = datetime.datetime.now()
    return f"📅 오늘은 {now.strftime('%Y년 %m월 %d일')}"

def get_news():
    return "📰 오늘의 뉴스: OpenAI, GPT-5 출시 예정!"

def show_info(user_id):
    info = user_data[user_id]["info"]
    print(f"\n👤 사용자: {user_id}")
    if "날씨" in info:
        print(get_weather())
    if "캘린더" in info:
        print(get_calendar())
    if "뉴스" in info:
        print(get_news())
    print("\n--- [단축키] ---\n[u]: 설정 수정   [n]: 새로운 사용자 등록   [ESC]: 종료")

def select_user_info():
    options = ["날씨", "캘린더", "뉴스"]
    print("\n✅ 표시할 정보를 선택하세요 (쉼표로 구분):")
    for idx, opt in enumerate(options, 1):
        print(f"{idx}. {opt}")
    choice = input("입력 (예: 1,3): ")
    selected = []
    for idx in choice.split(","):
        try:
            selected.append(options[int(idx) - 1])
        except:
            pass
    return selected

# --- 사용자 등록 ---
def register_new_user():
    new_id = input("\n🆕 새로운 사용자 ID 입력: ")
    save_path = os.path.join(FACES_DIR, new_id)
    os.makedirs(save_path, exist_ok=True)

    print("😄 얼굴 데이터를 수집합니다. 정면을 바라보세요...")
    cap = cv2.VideoCapture(0)
    count = 0
    while True:
        ret, frame = cap.read()
        if not ret: break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x,y,w,h) in faces:
            roi = gray[y:y+h, x:x+w]
            cv2.imwrite(os.path.join(save_path, f"{count}.png"), roi)
            count += 1
            cv2.rectangle(frame, (x,y), (x+w,y+h), (255,0,0), 2)
            cv2.putText(frame, f"Count: {count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

        cv2.imshow("Registering...", frame)
        if cv2.waitKey(1) == 27 or count >= 100:
            break

    cap.release()
    cv2.destroyAllWindows()

    user_data[new_id] = {"info": select_user_info()}
    save_user_data()
    train_model()

# --- 모델 학습 ---
def train_model():
    faces = []
    labels = []
    for user_id in os.listdir(FACES_DIR):
        if not os.path.isdir(os.path.join(FACES_DIR, user_id)):
            continue
        for file in os.listdir(os.path.join(FACES_DIR, user_id)):
            img_path = os.path.join(FACES_DIR, user_id, file)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is not None:
                faces.append(img)
                labels.append(int(hash(user_id) % 100000))  # 해시값을 정수 ID로 사용

    if faces:
        recognizer.train(faces, np.array(labels))
        recognizer.write(MODEL_PATH)
        print("✅ 모델 학습 완료")

# --- 사용자 인식용 해시 매핑 ---
def get_user_from_hash(h):
    for uid in user_data:
        if int(hash(uid) % 100000) == h:
            return uid
    return None

# --- 실행 시작 ---
if os.path.exists(MODEL_PATH):
    recognizer.read(MODEL_PATH)
else:
    print("⚠️ 학습된 모델이 없습니다. 사용자 등록을 먼저 진행하세요.")

cap = cv2.VideoCapture(0)
current_user = None
detected_hash = None

print("\n[스마트미러 시스템 시작]")
print("[단축키] n: 새 사용자 등록 | u: 설정 수정 | ESC: 종료\n")

while True:
    ret, frame = cap.read()
    if not ret: break
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x,y,w,h) in faces:
        roi = gray[y:y+h, x:x+w]
        try:
            label, confidence = recognizer.predict(roi)
            user_id = get_user_from_hash(label)

            if user_id and user_id != current_user:
                current_user = user_id
                show_info(current_user)

            cv2.putText(frame, f"{user_id}", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
            cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), 2)
        except:
            pass

    cv2.imshow("Smart Mirror", frame)
    key = cv2.waitKey(1)

    if key == ord('n'):
        register_new_user()
        recognizer.read(MODEL_PATH)
        current_user = None
    elif key == ord('u') and current_user:
        print(f"\n⚙️ [{current_user}] 설정 변경:")
        user_data[current_user]["info"] = select_user_info()
        save_user_data()
    elif key == 27:
        break

cap.release()
cv2.destroyAllWindows()