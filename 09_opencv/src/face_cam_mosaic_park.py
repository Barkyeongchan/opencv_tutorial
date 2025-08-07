import cv2
import dlib

# 얼굴 검출기와 랜드마크 검출기 생성
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('./shape_predictor_68_face_landmarks.dat')

cap = cv2.VideoCapture(0)
rate = 15  # 모자이크 비율

while cap.isOpened():
    ret, img = cap.read()
    if not ret:
        print('no frame.');break
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # 얼굴 영역 검출
    faces = detector(gray)

    for rect in faces:

        # 얼굴 영역을 좌표로 변환 후 사각형 표시
        x,y = rect.left(), rect.top()
        w,h = rect.right()-x, rect.bottom()-y

        roi = img[y:y+h, x:x+w]
        if roi.size == 0:
            continue
        roi = cv2.resize(roi, (w // rate, h // rate))
        roi = cv2.resize(roi, (w, h), interpolation=cv2.INTER_AREA)
        img[y:y+h, x:x+w] = roi
    
    cv2.imshow("mosaic", img)
    if cv2.waitKey(1)== 27:
        break

cv2.destroyAllWindows()
cap.release()