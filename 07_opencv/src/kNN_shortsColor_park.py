import cv2
import numpy as np
import pandas as pd
import os
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import pickle
from PIL import ImageFont, ImageDraw, Image

# --- 설정 ---
CSV_PATH = 'color_dataset.csv'
MODEL_PATH = 'knn_model.pkl'
roi_size = 100
mode = 'predict'  # 'learn' or 'predict'
knn_k = 5
label_encoder = LabelEncoder()
color_list = []

# 한글 폰트 경로 (윈도우 Malgun Gothic)
FONT_PATH = 'C:/Windows/Fonts/malgun.ttf'  

# --- PIL을 이용한 한글 출력 함수 ---
def put_text_korean(img, text, pos, font_path=FONT_PATH, font_size=30, color=(255,255,255)):
    # OpenCV 이미지를 PIL 이미지로 변환 (BGR->RGB)
    img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)
    font = ImageFont.truetype(font_path, font_size)
    draw.text(pos, text, font=font, fill=color)
    # 다시 PIL->OpenCV 이미지(BGR)로 변환
    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

# --- KNN 함수 ---
def knn_predict(X_train, y_train, x, k):
    distances = np.linalg.norm(X_train - x, axis=1)
    nearest = np.argsort(distances)[:k]
    top_k_labels = y_train[nearest]
    most_common = Counter(top_k_labels).most_common()
    pred_label = most_common[0][0]
    confidence = most_common[0][1] / k
    return pred_label, confidence

# --- 학습 함수 ---
def train_model():
    global label_encoder
    if not os.path.exists(CSV_PATH):
        print("❌ 학습 데이터가 없습니다.")
        return None, None, None
    df = pd.read_csv(CSV_PATH)
    X = df[['R', 'G', 'B']].values / 255.0
    y = df['label'].values

    # 라벨 인코더 학습
    label_encoder.fit(y)
    y_encoded = label_encoder.transform(y)

    X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)
    acc = 0
    best_k = knn_k
    for k in [3, 5, 7, 9]:
        correct = 0
        for i in range(len(X_test)):
            pred, _ = knn_predict(X_train, y_train, X_test[i], k)
            if pred == y_test[i]:
                correct += 1
        accuracy = correct / len(X_test)
        if accuracy > acc:
            acc = accuracy
            best_k = k
    print(f"✅ 최적 K: {best_k}, 정확도: {acc*100:.2f}%")
    with open(MODEL_PATH, 'wb') as f:
        pickle.dump((X_train, y_train, best_k), f)
    return X_train, y_train, best_k

# --- 마우스 콜백: ROI 위치 이동 ---
cx, cy = 320, 240
def mouse_callback(event, x, y, flags, param):
    global cx, cy
    if event == cv2.EVENT_LBUTTONDOWN:
        cx, cy = x, y

# --- 메인 ---
cv2.namedWindow("Color Recognizer")
cv2.setMouseCallback("Color Recognizer", mouse_callback)

# --- 모델 로딩 ---
if os.path.exists(MODEL_PATH):
    with open(MODEL_PATH, 'rb') as f:
        X_train, y_train, best_k = pickle.load(f)
    # label_encoder도 데이터셋 기준으로 다시 학습
    if os.path.exists(CSV_PATH):
        df = pd.read_csv(CSV_PATH)
        label_encoder.fit(df['label'])
else:
    X_train, y_train, best_k = train_model()
    if X_train is None:
        X_train, y_train, best_k = np.empty((0, 3)), np.empty((0,)), knn_k

cap = cv2.VideoCapture(0)
print("🎥 웹캠 실행 중. 'L': 학습 모드, 'P': 예측 모드, 'S': 모델 저장, 'R': 데이터 초기화, 'Q' 또는 ESC: 종료")

while True:
    ret, frame = cap.read()
    if not ret:
        break
    h, w, _ = frame.shape
    x1, y1 = cx - roi_size // 2, cy - roi_size // 2
    x2, y2 = cx + roi_size // 2, cy + roi_size // 2
    roi = frame[y1:y2, x1:x2]

    if roi.shape[0] > 0 and roi.shape[1] > 0:
        avg_color = roi.mean(axis=0).mean(axis=0)
        norm_color = avg_color / 255.0

        if mode == 'predict' and len(X_train) > 0:
            pred, conf = knn_predict(X_train, y_train, norm_color, best_k)
            # 숫자가 아닌 문자열 라벨로 변환
            try:
                color_name = label_encoder.inverse_transform([int(pred)])[0]
            except Exception:
                color_name = str(pred)
            frame = put_text_korean(frame, f"{color_name} ({conf*100:.1f}%)", (10, 50), font_size=30, color=(0,255,0))

        elif mode == 'learn':
            frame = put_text_korean(frame, "학습 모드: 숫자키(1~7)로 라벨링", (10, 50), font_size=30, color=(0,0,255))
            labels_info = ["Red = 1", "Blue = 2", "Green = 3", "Yellow = 4", "Black = 5", "White = 6", "Gray = 7"]
            for i, text in enumerate(labels_info):
                y_pos = 50 + 35 * (i + 1)  # 위 문장 밑으로 한 줄씩
                frame = put_text_korean(frame, text, (10, y_pos), font_size=25, color=(0,0,255))
        
        # 색상 히스토리 시각화
        color_list.append(avg_color)
        if len(color_list) > 10:
            color_list.pop(0)
        for i, c in enumerate(color_list):
            cv2.rectangle(frame, (10+i*30, h-40), (30+i*30, h-10), c.astype(int).tolist(), -1)

    # ROI 사각형 표시
    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 255), 2)
    frame = put_text_korean(frame, f"MODE: {mode.upper()}", (10, 10), font_size=20, color=(255, 255, 0))

    # 프레임 출력
    cv2.imshow("Color Recognizer", frame)
    key = cv2.waitKey(1) & 0xFF

    if key == ord('q') or key == 27:
        break
    elif key == ord('l'):
        mode = 'learn'
    elif key == ord('p'):
        mode = 'predict'
    elif key == ord('s'):
        train_model()
    elif key == ord('r'):
        if os.path.exists(CSV_PATH):
            os.remove(CSV_PATH)
        print("🔄 데이터셋 초기화 완료.")
        X_train, y_train = np.empty((0, 3)), np.empty((0,))
    elif mode == 'learn' and ord('1') <= key <= ord('7'):
        label = str(key - ord('0'))
        r, g, b = avg_color.astype(int)
        print(f"➕ 샘플 추가: {r}, {g}, {b}, 라벨: {label}")
        new_data = pd.DataFrame([[r, g, b, label]], columns=['R', 'G', 'B', 'label'])
        if os.path.exists(CSV_PATH):
            df = pd.read_csv(CSV_PATH)
            df = pd.concat([df, new_data], ignore_index=True)
        else:
            df = new_data
        df.to_csv(CSV_PATH, index=False)

cap.release()
cv2.destroyAllWindows()
