import cv2
import numpy as np
import pandas as pd
import os
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import pickle
from PIL import ImageFont, ImageDraw, Image

# --- 설정 변수 ---
CSV_PATH = 'color_dataset.csv'    # 학습 데이터 파일명
MODEL_PATH = 'knn_model.pkl'      # 저장할 KNN 모델 파일명
roi_size = 100                   # 관심영역(ROI) 크기 (정사각형 한 변 길이)
mode = 'predict'                 # 프로그램 시작 모드 ('learn' or 'predict')
knn_k = 5                       # KNN의 k값 (이웃 개수)
label_encoder = LabelEncoder()  # 문자열 라벨을 숫자로 변환할 때 사용
color_list = []                 # 최근 색상 기록 저장용 리스트

# 윈도우 기본 한글 폰트 경로 (필요 시 변경)
FONT_PATH = 'C:/Windows/Fonts/malgun.ttf'  

# --- 한글 출력 함수 (OpenCV는 한글 지원이 약해서 PIL 사용) ---
def put_text_korean(img, text, pos, font_path=FONT_PATH, font_size=30, color=(255,255,255)):
    # OpenCV(BGR) 이미지를 PIL(RGB) 이미지로 변환
    img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)
    font = ImageFont.truetype(font_path, font_size)   # 폰트 및 크기 설정
    draw.text(pos, text, font=font, fill=color)       # 텍스트 그리기
    # 다시 PIL 이미지를 OpenCV 형식(BGR)으로 변환 후 반환
    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

# --- KNN 예측 함수 ---
def knn_predict(X_train, y_train, x, k):
    distances = np.linalg.norm(X_train - x, axis=1)  # 각 학습 데이터와 거리 계산
    nearest = np.argsort(distances)[:k]              # 가장 가까운 k개 데이터 인덱스
    top_k_labels = y_train[nearest]                   # 그 데이터들의 라벨
    most_common = Counter(top_k_labels).most_common() # 최빈값 찾기
    pred_label = most_common[0][0]                    # 예측 라벨
    confidence = most_common[0][1] / k                 # 신뢰도(빈도/k)
    return pred_label, confidence

# --- 모델 학습 함수 ---
def train_model():
    global label_encoder
    if not os.path.exists(CSV_PATH):  # 학습 데이터 파일이 없으면 종료
        print("❌ 학습 데이터가 없습니다.")
        return None, None, None

    df = pd.read_csv(CSV_PATH)  # CSV에서 데이터 읽기
    X = df[['R', 'G', 'B']].values / 255.0  # RGB 값 0~1로 정규화
    y = df['label'].values                   # 문자열 라벨

    label_encoder.fit(y)             # 문자열 라벨 → 숫자 인코딩 학습
    y_encoded = label_encoder.transform(y)

    # 학습/검증 데이터 분리 (랜덤 시드 고정)
    X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

    acc = 0
    best_k = knn_k
    # 여러 k값에 대해 가장 좋은 정확도 찾기
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

    # 학습 결과 저장 (X_train, y_train, best_k)
    with open(MODEL_PATH, 'wb') as f:
        pickle.dump((X_train, y_train, best_k), f)

    return X_train, y_train, best_k

# --- ROI 위치 조정을 위한 마우스 콜백 함수 ---
cx, cy = 320, 240  # 초기 ROI 중심 위치 (프레임 중간)
def mouse_callback(event, x, y, flags, param):
    global cx, cy
    if event == cv2.EVENT_LBUTTONDOWN:
        cx, cy = x, y   # 클릭 위치로 ROI 중심 이동

# --- 메인 프로그램 시작 ---
cv2.namedWindow("Color Recognizer")
cv2.setMouseCallback("Color Recognizer", mouse_callback)

# 모델 파일이 있으면 불러오고, 없으면 학습 실행
if os.path.exists(MODEL_PATH):
    with open(MODEL_PATH, 'rb') as f:
        X_train, y_train, best_k = pickle.load(f)
    # 학습 데이터에 맞춰 라벨 인코더 다시 학습
    if os.path.exists(CSV_PATH):
        df = pd.read_csv(CSV_PATH)
        label_encoder.fit(df['label'])
else:
    X_train, y_train, best_k = train_model()
    if X_train is None:
        # 학습 데이터가 없으면 빈 배열 세팅
        X_train, y_train, best_k = np.empty((0, 3)), np.empty((0,)), knn_k

cap = cv2.VideoCapture(0)
print("🎥 웹캠 실행 중. 'L': 학습 모드, 'P': 예측 모드, 'S': 모델 저장, 'R': 데이터 초기화, 'Q' 또는 ESC: 종료")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    h, w, _ = frame.shape
    # ROI 좌표 계산 (중심 기준)
    x1, y1 = cx - roi_size // 2, cy - roi_size // 2
    x2, y2 = cx + roi_size // 2, cy + roi_size // 2
    roi = frame[y1:y2, x1:x2]  # ROI 영역 추출

    if roi.shape[0] > 0 and roi.shape[1] > 0:
        avg_color = roi.mean(axis=0).mean(axis=0)  # ROI 내 평균 BGR 색상 계산
        norm_color = avg_color / 255.0             # 0~1 범위로 정규화

        if mode == 'predict' and len(X_train) > 0:
            # 예측 모드일 때 KNN으로 색상 예측
            pred, conf = knn_predict(X_train, y_train, norm_color, best_k)
            try:
                color_name = label_encoder.inverse_transform([int(pred)])[0]  # 숫자→문자 라벨 변환
            except Exception:
                color_name = str(pred)
            frame = put_text_korean(frame, f"{color_name} ({conf*100:.1f}%)", (10, 50), font_size=30, color=(0,255,0))

        elif mode == 'learn':
            # 학습 모드 안내 텍스트 출력 + 키별 색상 안내 표시
            frame = put_text_korean(frame, "학습 모드: 숫자키(1~7)로 라벨링", (10, 50), font_size=30, color=(0,0,255))
            labels_info = ["Red = 1", "Blue = 2", "Green = 3", "Yellow = 4", "Black = 5", "White = 6", "Gray = 7"]
            for i, text in enumerate(labels_info):
                y_pos = 50 + 35 * (i + 1)
                frame = put_text_korean(frame, text, (10, y_pos), font_size=25, color=(0,0,255))
        
        # 최근 색상 히스토리 사각형으로 시각화
        color_list.append(avg_color)
        if len(color_list) > 10:
            color_list.pop(0)
        for i, c in enumerate(color_list):
            cv2.rectangle(frame, (10+i*30, h-40), (30+i*30, h-10), c.astype(int).tolist(), -1)

    # ROI 사각형 표시
    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 255), 2)
    frame = put_text_korean(frame, f"MODE: {mode.upper()}", (10, 10), font_size=20, color=(255, 255, 0))

    # 화면에 프레임 출력
    cv2.imshow("Color Recognizer", frame)

    # 키 입력 처리
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q') or key == 27:   # 'q' 또는 ESC 키 종료
        break
    elif key == ord('l'):
        mode = 'learn'                  # 학습 모드 전환
    elif key == ord('p'):
        mode = 'predict'               # 예측 모드 전환
    elif key == ord('s'):
        train_model()                  # 모델 재학습 및 저장
    elif key == ord('r'):
        # 데이터셋 초기화
        if os.path.exists(CSV_PATH):
            os.remove(CSV_PATH)
        print("🔄 데이터셋 초기화 완료.")
        X_train, y_train = np.empty((0, 3)), np.empty((0,))
    elif mode == 'learn' and ord('1') <= key <= ord('7'):
        # 학습 모드에서 숫자키 입력 시 샘플 저장
        label = str(key - ord('0'))    # 키 값 → 문자열 라벨 변환
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