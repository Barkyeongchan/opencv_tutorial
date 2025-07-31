# 자동차 번호판 추출

import cv2
import numpy as np

# @변수 정의
car_plate01 = cv2.imread('../img/car_01')
car_plate02 = cv2.imread('../img/car_02')
car_plate03 = cv2.imread('../img/car_03')
car_plate04 = cv2.imread('../img/car_04')
car_plate05 = cv2.imread('../img/car_05')

win_name = "License Plate Extractor"
rows, cols = car_plate01.shape[:2]
draw = car_plate01.copy()
pts_cnt = 0
pts = np.zeros((4,2), dtype=np.float32)

# @마우스 이벤트 함수
def onMouse(event, x, y, flags, param):  # 마우스 이벤트 콜백 함수 구현
    global  pts_cnt                      # 마우스로 찍은 좌표의 갯수 저장
    if event == cv2.EVENT_LBUTTONDOWN:   # 마우스 왼쪽 버튼 클릭시
        # 1. 클릭 지점에 원 그리기
        cv2.circle(draw, (x,y), 5, (0,255,0), -1)  # 클릭 지점에 녹색 5px의 원 그리기
        cv2.imshow(win_name, draw)

