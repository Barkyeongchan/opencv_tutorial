import cv2
import matplotlib.pyplot as plt
import pyzbar.pyzbar as pyzbar
import webbrowser

# @카메라 캡쳐 활성화
cap = cv2.VideoCapture(0)

# @웹 사이트 이동 횟수 제한 조건
link_opened = False

# @카메라 캡쳐 조건 추가
while (cap.isOpened()):     # 카메라 캡쳐가 열려있는 동안 
    ret, img = cap.read()

    if not ret:
        continue

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    decoded = pyzbar.decode(gray)

    # @QR코드의 데이터와 형식 출력
    for d in decoded:
        x, y, w, h = d.rect     # QR코드의 x, y, w, h 값은 d.rect에 저장됨
        barcode_data = d.data.decode('utf-8')
        barcode_type = d.type

        # @웹 사이트 한 번 만 열기
        if not link_opened and barcode_data.startswith("http"):
            webbrowser.open(barcode_data)
            link_opened = True

        text = '%s (%s)' % (barcode_data, barcode_type) # 바코드 데이터와 형식을 text에 저장

        # @QR을 인식하는 사각형 그리기
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        # @QR옆에 text 넣기
        cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)

    cv2.imshow('camera', img)

    key = cv2.waitKey(1)
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()