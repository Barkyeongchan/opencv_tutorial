# BackgroundSubtractorMOG로 배경 제거

import numpy as np, cv2

#cap = cv2.VideoCapture('../img/walking.avi')
cap = cv2.VideoCapture(0)
fps = cap.get(cv2.CAP_PROP_FPS) # 프레임 수 구하기
delay = int(1000/fps)

# 배경 제거 객체 생성
# history : 과거 프레임의 객수, 배경을 학습하는데 얼마나 많으 프레임을 기억할지 정함
# varThreshold : 픽셀이 객체인지 배경인지 구분하는 기준
fgbg = cv2.createBackgroundSubtractorMOG2(50, 45, detectShadows=False)
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    # 배경 제거 마스크 계산 --- ②
    fgmask = fgbg.apply(frame)
    cv2.imshow('frame',frame)
    cv2.imshow('bgsub',fgmask)
    if cv2.waitKey(1) & 0xff == 27:
        break

cap.release()
cv2.destroyAllWindows()