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

cv2.waitKey(0)
cv2.destroyAllWindows()