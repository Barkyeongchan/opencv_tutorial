import cv2
import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

# 기본 색상 레이블 (숫자 키와 매핑)
color_labels = {
    49: "Red",    # '1'
    50: "Blue",   # '2'
    51: "Green",  # '3'
    52: "Yellow", # '4'
    53: "Black",  # '5'
    54: "White",  # '6'
    55: "Gray"    # '7'
}

# 수집 데이터 저장용 리스트 (B, G, R, label)
samples = []

# ROI 설정 (중앙, 크기 100x100)
roi_size = 100
frame_width, frame_height = 640, 480
cx, cy = frame_width // 2, frame_height // 2

# 학습용 KNN 모델 및 스케일러 (초기엔 None)
knn_model = None
scaler = None

def load_dataset_and_train():
    global knn_model, scaler
    try:
        df = pd.read_csv('color_dataset.csv')
        X = df[['B', 'G', 'R']].values.astype(float)
        y = df['Label'].values

        scaler = MinMaxScaler()
        X_scaled = scaler.fit_transform(X)

        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=0.2, random_state=42)

        knn_model = KNeighborsClassifier(n_neighbors=3)
        knn_model.fit(X_train, y_train)

        acc = knn_model.score(X_test, y_test)
        print(f"K-NN 모델 학습 완료, 테스트 정확도: {acc*100:.2f}%")
        return True
    except Exception as e:
        print("데이터셋 없음 또는 학습 실패:", e)
        return False

# 웹캠 열기
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, frame_width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_height)

print("웹캠 실행 중. ESC 키로 종료")
print("마우스 왼쪽 클릭으로 ROI 내 색상 샘플 수집 후 1~7 숫자키로 라벨링")

current_color = None
click_pos = None
current_frame = None

def mouse_callback(event, x, y, flags, param):
    global current_color, click_pos
    if event == cv2.EVENT_LBUTTONDOWN:
        if current_frame is None:
            return
        if (cx - roi_size//2 <= x <= cx + roi_size//2) and (cy - roi_size//2 <= y <= cy + roi_size//2):
            # ROI 내 클릭했으면 평균 색상 추출
            roi = frame[cy - roi_size//2:cy + roi_size//2, cx - roi_size//2:cx + roi_size//2]
            avg_color = np.mean(roi.reshape(-1,3), axis=0).astype(int)
            current_color = avg_color
            click_pos = (x, y)
            print(f"샘플 색상 추출됨: BGR = {avg_color}")

cv2.namedWindow("Color Collect & Predict")
cv2.setMouseCallback("Color Collect & Predict", mouse_callback)

model_ready = load_dataset_and_train()

while True:
    ret, frame = cap.read()
    if not ret:
        break
    current_frame = frame.copy()

    # ROI 사각형 표시
    x1, y1 = cx - roi_size//2, cy - roi_size//2
    x2, y2 = cx + roi_size//2, cy + roi_size//2
    cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 2)

    # 현재 샘플 색상 표시
    if current_color is not None:
        cv2.putText(frame, f"Sampled BGR: {current_color}", (10,30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
        cv2.circle(frame, click_pos, 5, (0,255,0), -1)

    # 예측 (모델 준비되면)
    if model_ready:
        roi = frame[y1:y2, x1:x2]
        avg_color = np.mean(roi.reshape(-1,3), axis=0).astype(int)
        avg_scaled = scaler.transform([avg_color])
        pred_label = knn_model.predict(avg_scaled)[0]
        cv2.putText(frame, f"Predicted Color: {pred_label}", (10, frame_height - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2)

    cv2.imshow("Color Collect & Predict", frame)

    key = cv2.waitKey(1)

    if key == 27:  # ESC 종료
        break
    elif key in color_labels and current_color is not None:
        label = color_labels[key]
        b, g, r = current_color
        samples.append([int(b), int(g), int(r), label])
        print(f"샘플 수집됨: {label} - BGR({b},{g},{r})")
        current_color = None

    elif key == ord('s'):
        # 수집한 샘플 저장
        if samples:
            df = pd.DataFrame(samples, columns=['B', 'G', 'R', 'Label'])
            df.to_csv('color_dataset.csv', index=False)
            print(f"샘플 {len(samples)}개 저장됨 (color_dataset.csv)")
            model_ready = load_dataset_and_train()
        else:
            print("저장할 샘플이 없습니다")

    # ROI 위치 조절 (화살표 키)
    elif key == 81:  # 왼쪽
        cx = max(cx - 10, roi_size//2)
    elif key == 83:  # 오른쪽
        cx = min(cx + 10, frame_width - roi_size//2)
    elif key == 82:  # 위
        cy = max(cy - 10, roi_size//2)
    elif key == 84:  # 아래
        cy = min(cy + 10, frame_height - roi_size//2)

cap.release()
cv2.destroyAllWindows()