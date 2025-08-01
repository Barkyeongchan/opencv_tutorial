# 검은색 라인이 그려진 종이 감지 시 촬영
# 히스토그램으로 픽셀 분석

import cv2
import numpy as np
import os
import matplotlib.pylab as plt

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# 촬영된 사진 저장 위치 설정
save_dir = '../result_screenshot'
os.makedirs(save_dir, exist_ok=True)
line_detected = False

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("카메라 프레임이 잡히지 않았습니다.")
        break

    # 그레이스케일로 전환
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # 종이를 구별하기 위한 쓰레싱
    _, paper_mask = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)

    # 종이의 윤곽선 검별
    contours, _ = cv2.findContours(paper_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    paper_present = False
    for cnt in contours:
        area = cv2.contourArea(cnt)
        # 카메라 범위 설정 (종이 감지를 위한 최소한의 크기)
        if area > 50000:
            paper_present = True
            x, y, w, h = cv2.boundingRect(cnt)
            cv2.rectangle(frame, (x,y), (x+w, y+h), (0,255,0), 2)

            # 종이에 그려진 검은색 라인을 찾기 위한 크롭
            paper_region = gray[y:y+h, x:x+w]
            _, line_mask = cv2.threshold(paper_region, 60, 255, cv2.THRESH_BINARY_INV)

            black_pixels = cv2.countNonZero(line_mask)
            # 검은색 라인 감지
            if black_pixels > 5000:
                # 여러 장 사진 찍기 방지용
                if not line_detected:
                    img_path = os.path.join(save_dir, f'pic.jpg')
                    cv2.imwrite(img_path, frame)
                    print("사진이 저장되었습니다.")
                    line_detected = True
            else:
                line_detected = False

    cv2.imshow("Pic", frame)

    if cv2.waitKey(1) == ord('q'):
        break

img = cv2.imread('../result_screenshot/pic.jpg')
cv2.imshow('img', img)

# 히스토그램 계산 및 그리기
channels = cv2.split(img)
colors = ('b', 'g', 'r')
for (ch, color) in zip (channels, colors):
    hist = cv2.calcHist([ch], [0], None, [256], [0, 256])
    plt.plot(hist, color = color)
plt.show()

cap.release()
cv2.destroyAllWindows()