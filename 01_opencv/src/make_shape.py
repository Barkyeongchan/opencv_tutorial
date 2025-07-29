import cv2
import numpy as np

space = np.zeros((1000, 1000), dtype=np.uint8)
line_color = 255

# @선 긋기
#space = cv2.line(space, (100, 100), (800, 400), line_color, 3, 1)

# @원 만들기
#space = cv2.circle(space, (600, 200), 100, line_color, 4, 1)

# @네모 만들기
#space = cv2.rectangle(space, (500, 200), (800, 400), line_color, 5, 1)

# @반원 만들기
#space = cv2.ellipse(space, (500, 300), (300, 200), 0, 90, 250, line_color, 5)

# @색 채우기
#obj1 = np.array([[300, 500], [500, 500], [400, 600], [200, 600]])
#obj2 = np.array([[600, 500], [800, 500], [700, 200]])
#color_fill = cv2.polylines(space, [obj1], True, line_color, 3)
#color_fill = cv2.fillPoly(space, [obj2], line_color, 1)

# @격자 만들기
grid_spacing = 25   #격자 간격

for x in range(0, space.shape[1], grid_spacing):
    cv2.line(space, (x, 0), (x, space.shape[0]), line_color, 1)

for y in range(0, space.shape[0], grid_spacing):
    cv2.line(space, (0, y), (space.shape[1], y), line_color, 1)

cv2.imshow('line', space)

cv2.waitKey(0)
cv2.destroyAllWindows()