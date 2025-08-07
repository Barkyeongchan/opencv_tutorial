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
