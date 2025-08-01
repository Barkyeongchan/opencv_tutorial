import cv2

# @카메라 캡쳐 실행
cap = cv2.VideoCapture(0)

# @해상도 변경
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

if cap.isOpened():
    while True:
        ret, img = cap.read()
        if ret: 
            cv2.imshow('camera', img)
            key = cv2.waitKey(1) & 0xFF    # 키 입력 대기
            if key == ord('q'):            # 'q'입력시 창 꺼짐
                break
        else:
            print('no frame')
            break
else:
    print("can't open camera.")

cap.release()
cv2.destroyAllWindows()