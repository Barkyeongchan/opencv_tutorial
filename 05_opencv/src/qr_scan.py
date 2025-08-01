import cv2
import matplotlib.pyplot as plt
import pyzbar.pyzbar as pyzbar

# @이미지 불러오기
img = cv2.imread('../img/frame.png')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)    # 이미지 그레이스케일

# @결과 출력
#plt.imshow(img)
plt.imshow(gray, cmap='gray')   # 맷플롯에서 그레이로 정의 필요
plt.show()

# @디코딩(pyzbar)
decoded = pyzbar.decode(gray)
print(decoded)

# @QR코드의 데이터와 형식 출력
for d in decoded:
    print(d.data.decode('utf-8'))
    print(d.type)

    # @QR인식을 위한 사각형 그리기
    #cv2.rectangle(img, ())
'''
img: 사각형을 그릴 이미지입니다.
pt1: 사각형의 왼쪽 상단 꼭지점 좌표입니다. (x, y) 형식의 튜플이어야 합니다.
pt2: 사각형의 오른쪽 하단 꼭지점 좌표입니다. (x, y) 형식의 튜플이어야 합니다.
color: 사각형의 색상입니다. (B, G, R) 형식의 튜플이나 스칼라 값으로 지정할 수 있습니다.
thickness: 선택적으로 사각형의 선 두께를 지정합니다. 기본값은 1입니다. 음수 값을 전달하면 내부를 채웁니다.
lineType: 선택적으로 선의 형태를 지정합니다. 기본값은 cv2.LINE_8입니다.
shift: 선택적으로 좌표값의 소수 부분을 비트 시프트할 양을 지정합니다.
'''

cv2.waitKey(0)
cv2.destroyAllWindows()