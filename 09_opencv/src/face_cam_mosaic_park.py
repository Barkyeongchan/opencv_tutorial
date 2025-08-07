import cv2
import dlib

# 얼굴 검출기와 랜드마크 검출기 생성
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('./shape_predictor_68_face_landmarks.dat')

cap = cv2.VideoCapture(0)
#cap.set(cv2.cv2.CAP_PROP_FRAME_WIDTH, 480)
#cap.set(cv2.cv2.CAP_PROP_FRAME_HEIGHT, 320)

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
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 1)
    
    cv2.imshow("face landmark", img)
    if cv2.waitKey(1)== 27:
        break

rate = 15               # 모자이크에 사용할 축소 비율 (1/rate)
win_title = 'mosaic'    # 창 제목
img = cv2.imread('../img/like_lenna.png')    # 이미지 읽기

while True:
    x,y,w,h = cv2.selectROI(win_title, img, False) # 관심영역 선택
    if w and h:
        roi = img[y:y+h, x:x+w]   # 관심영역 지정
        roi = cv2.resize(roi, (w//rate, h//rate)) # 1/rate 비율로 축소
        # 원래 크기로 확대
        roi = cv2.resize(roi, (w,h), interpolation=cv2.INTER_AREA)  
        img[y:y+h, x:x+w] = roi   # 원본 이미지에 적용
        cv2.imshow(win_title, img)
    else:
        break
    
cv2.destroyAllWindows()
cap.release()