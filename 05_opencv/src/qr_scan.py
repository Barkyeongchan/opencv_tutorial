import cv2
import matplotlib.pyplot as plt
import pyzbar.pyzbar as pyzbar

# @이미지 불러오기
#img = cv2.imread('../img/frame.png')
#gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)    # 이미지 그레이스케일
cap = cv2.VideoCapture(0)   # 비디오 캡쳐 활성화

# @이미지 캡쳐 조건 추가
while (cap.isOpened()):     # 카메라 캡쳐가 열려있으면 
    ret, img = cap.read()

    if not ret:
        continue

    #img = cv2.imread('../img/frame.png')
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    decoded = pyzbar.decode(gray)

    # @QR코드의 데이터와 형식 출력
    for d in decoded:
        x, y, w, h = d.rect

        barcode_data = d.data.decode('utf-8')
        barcode_type = d.type
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)


        #print(d.data.decode('utf-8'))
        #barcode_data = d.data.decode('utf-8')
        #print(d.type)   
        #barcode_type = d.type

        text = '%s (%s)' % (barcode_data, barcode_type) # 바코드 데이터와 형식을 text에 저장

        # @QR을 인식하는 사각형 그리기
        #cv2.rectangle(img, (d.rect[0], d.rect[1]), (d.rect[0] + d.rect[2], d.rect[1] + d.rect[3]), (0, 255, 0), 20)
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        # @QR에 글자 넣기
        #cv2.putText(img, text, (d.rect[0], d.rect[3] + 450), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 0), 5, cv2.LINE_AA)
        cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)

    cv2.imshow('camera', img)

    key = cv2.waitKey(1)
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

# @테스트 결과 출력
#plt.imshow(img)
#plt.imshow(gray, cmap='gray')   # 맷플롯에서 그레이로 정의 필요
#plt.show()

## @디코딩(pyzbar)
#decoded = pyzbar.decode(gray)
#print(decoded)
#
## @QR코드의 데이터와 형식 출력
#for d in decoded:
#    print(d.data.decode('utf-8'))
#    barcode_data = d.data.decode('utf-8')   
#    print(d.type)
#    barcode_type = d.type
#
#    text = '%s (%s)' % (barcode_data, barcode_type) # 바코드 데이터와 형식을 text에 저장
#
#    # @QR을 인식하는 사각형 그리기
#    cv2.rectangle(img, (d.rect[0], d.rect[1]), (d.rect[0] + d.rect[2], d.rect[1] + d.rect[3]),\
#                  (0, 255, 0), 20)
#    
#    # @QR에 글자 넣기
#    cv2.putText(img, text, (d.rect[0], d.rect[3] + 450), cv2.FONT_HERSHEY_SIMPLEX, 3,\
#                (0, 0, 0), 5, cv2.LINE_AA)
#
## @디코딩 이후 결과 출력
#plt.imshow(img)
#plt.show()
#
#cv2.waitKey(0)
#cv2.destroyAllWindows()