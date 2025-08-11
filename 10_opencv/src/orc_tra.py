import easyocr
import cv2
import matplotlib.pyplot as plt

reader = easyocr.Reader(['ch_tra', 'en'], gpu=False)
img_path = '../img/chinese_tra.jpg'
img = cv2.imread(img_path)
# 1. 이미지 불러오기 
#plt.figure(figsize=(8,8))
#plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
#plt.show()

# 2. 이미지 텍스트 인식 
result = reader.readtext(img_path)
print(result)

# 3. 인식된 텍스트 확인해보기 

THRESHOLD = 0.5

for bbox, text, conf in result:
    if conf >= THRESHOLD:
        print(text)
        cv2.rectangle(img, pt1=bbox[0], pt2=bbox[2], color=(0, 255, 0), thickness=2)
    plt.figure(figsize=(8,8))
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.show()

cv2.waitKey(0)
cv2.destroyAllWindows()