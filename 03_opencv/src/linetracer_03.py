# 검은색 라인 부분 관심영역 설정
# 실시간으로 인식

import cv2
import numpy as np

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

if cap.isOpened():
    while True:
        ret, frame = cap.read()
        if not ret:
            print('카메라 프레임 잡기 실패')
            break
        
        # ROI 설정
        height, width = frame.shape[:2]

        roi_width = int(width * 0.3) # 스크린 가운데
        roi_height = int(height * 0.3) # 30% 높이
        roi_x = int((width - roi_width) / 2)  # 가운데 정렬
        roi_y = int((height - roi_height) * 1)

        roi = frame[roi_y:roi_y + roi_height, roi_x:roi_x + roi_width]

        cv2.rectangle(frame, (roi_x, roi_y), (roi_x + roi_width, roi_y + roi_height), (0, 255, 0), 2)
        
        # ROI 영역을 그레이스케일로 전환
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

        # 검은색을 강조하기 위해 바이너리 이미지로 만들기
        _, thresh = cv2.threshold(gray, 60, 255, cv2.THRESH_BINARY_INV)

        # 검은색 윤곽선 찾기
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for cnt in contours:
            area = cv2.contourArea(cnt)
            # 너무 작은 부분은 필터링
            if area > 500:
                x, y, w, h = cv2.boundingRect(cnt)
                # 빨간색 사각형으로 강조
                cv2.rectangle(roi, (x, y), (x + w, y + h), (0, 0, 255), 1)
                cv2.putText(roi, "black area", (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        # 실시간 비디오로 보여주기
        cv2.imshow('Black Line Detection', frame)
        cv2.imshow('Threshold', thresh)

        if cv2.waitKey(1) == ord('q'):
            break
else:
    print("카메라를 열 수 없습니다.")

cap.release()
cv2.destroyAllWindows()