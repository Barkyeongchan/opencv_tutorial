# 자동차 번호판 추출

import cv2
import numpy as np
import datetime
import os

max_img = input('추출하려는 이미지 개수를 입력해 주세요.(최대 5): ')
for i in range(1, int(max_img)+1):
    # @변수 정의
    full_path = '../img/car_0' + str(i) +'.jpg'
    car_plate = cv2.imread(full_path)

    if car_plate is None:
        print("❌ car_plate01 이미지가 제대로 불러와지지 않았습니다.")
        exit()

    win_name = "License Plate Extractor"
    rows, cols = car_plate.shape[:2]
    draw = car_plate.copy()
    pts_cnt = 0
    pts = np.zeros((4,2), dtype=np.float32)

    # @마우스 이벤트 함수
    def onMouse(event, x, y, flags, param):  # 마우스 이벤트 콜백 함수 구현
        global  pts_cnt                      # 마우스로 찍은 좌표의 갯수 저장
        if event == cv2.EVENT_LBUTTONDOWN:   # 마우스 왼쪽 버튼 클릭시
            # 1. 클릭 지점에 원 그리기
            cv2.circle(draw, (x,y), 5, (0,255,0), -1)  # 클릭 지점에 녹색 5px의 원 그리기
            cv2.imshow(win_name, draw)

            # 2. 좌표 배열에 마우스 좌표 저장
            pts[pts_cnt] = [x,y]

            # 3. 카운터 증가
            pts_cnt+=1

            # 4. 4개 점 완성시 변환 실행
            if pts_cnt == 4:                       # 좌표가 4개 수집됨

                # @좌표 정렬 알고리즘 이해
                # 좌표 4개 중 상하좌우 찾기
                sm = pts.sum(axis=1)                 # 4쌍의 좌표 각각 x+y 계산
                diff = np.diff(pts, axis = 1)        # 4쌍의 좌표 각각 x-y 계산

                topLeft = pts[np.argmin(sm)]         # x+y가 가장 작은 값이 좌상단 좌표
                bottomRight = pts[np.argmax(sm)]     # x+y가 가장 큰 값이 우하단 좌표
                topRight = pts[np.argmin(diff)]      # x-y가 가장 작은 것이 우상단 좌표
                bottomLeft = pts[np.argmax(diff)]    # x-y가 가장 큰 값이 좌하단 좌표

                # 변환 전 4개 좌표 
                pts1 = np.float32([topLeft, topRight, bottomRight , bottomLeft])

                '''
                한국 번호판 표준 규격
                일반 승용차: 가로 335mm × 세로 170mm (약 2:1 비율)
                대형차량: 가로 440mm × 세로 220mm (2:1 비율)
                픽셀 변환: 300×150 또는 400×200 권장
                '''
                # 번호판 치수 계산 - 한국 표준 규격에 따른 2:1비율 사용
                width = 300
                height = 150

                # 변환 후 4개 좌표
                pts2 = np.float32([[0,0], [width-1,0], 
                                    [width-1,height-1], [0,height-1]])

                # 변환 행렬 계산 
                mtrx = cv2.getPerspectiveTransform(pts1, pts2)
                # 원근 변환 적용
                result = cv2.warpPerspective(car_plate, mtrx, (int(width), int(height)))

                # @파일 저장 기능 구현

                # 1. 저장 경로 처리
                save_dir = "../extracted_plates"   # 저장 폴더가 없으면 생성
                if not os.path.exists(save_dir):
                    os.makedirs(save_dir)

                # 2. 타임 스탬프 기반
                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                filename_time = f"../extracted_plates/plate_{timestamp}.png"    # png형식 선택

                # 3. 순번 기반
                existing_files = len(os.listdir(save_dir))
                filename_os = f"../extracted_plates/plate_{existing_files+1:03d}.png" # png형식 선택

                success = cv2.imwrite(filename_time, result) # 타임 스탬프 파일 저장
                if success:
                    print(f"번호판 저장 완료: {filename_time}")
                    cv2.imshow('Extracted Plate', result)
                else:
                    print("저장 실패!")

                success = cv2.imwrite(filename_os, result) # 순번 파일 저장
                if success:
                    print(f"번호판 저장 완료: {filename_os}")
                    cv2.imshow('Extracted Plate', result)
                else:
                    print("저장 실패!")

    cv2.imshow(win_name, car_plate)
    cv2.setMouseCallback(win_name, onMouse)

    cv2.waitKey(0)
    cv2.destroyAllWindows()