import cv2
import numpy as np

image = cv2.imread('../img/like_lenna.png')
lenna_img = cv2.resize(image, (500, 500))
color = (147, 20, 255)

# @격자 만들기
#grid_spacing = 25   #격자 간격
#for x in range(0, space.shape[1], grid_spacing):
#    cv2.line(lenna_img, (x, 0), (x, space.shape[0]), color, 1)
#
#for y in range(0, space.shape[0], grid_spacing):
#    cv2.line(lenna_img, (0, y), (space.shape[1], y), color, 1)

# @테두리선 그리기
line1 = np.array([[360,0], [395,60], [425, 65], [450, 70], [465, 75], [475, 85],
                  [475, 100],[450,140],[425,160],[410,170],[400,190],[405,210],
                  [410,225],[390,270],[395,280],[385,320],[380,340],[375,350],
                  [340,360],[320,390],[315,400],[335,445],[335,475],[350,500],
                  [35,500],[65,450],[120,400],[140,340],[150,340],[170,325],
                  [170,300],[160,230],[125,225],[100,200],[90,175],[100,150],
                  [125,125],[155,100],[180,25],[195,0]])

cv2.polylines(lenna_img, [line1], True, color, 3)

cv2.imshow('Image Window', lenna_img)

cv2.waitKey(0)
cv2.destroyAllWindows()