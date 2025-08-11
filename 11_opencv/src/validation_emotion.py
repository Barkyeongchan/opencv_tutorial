import cv2
import matplotlib.pyplot as plt

image = cv2.imread('../data/img/charles.jpg')

plt.figure(figsize=(8,8))
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.show()