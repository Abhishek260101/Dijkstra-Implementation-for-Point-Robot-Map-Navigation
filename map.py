import numpy as np
import matplotlib.pyplot as plt
import cv2

# Creating a map according to the given requirements
canvas = np.ones((500, 1200, 3), dtype=np.uint8) * 255
cv2.rectangle(canvas, pt1=(100, 500), pt2=(175, 100), color=(0, 0, 0), thickness=-1)
cv2.rectangle(canvas, pt1=(275, 400), pt2=(350, 0), color=(0, 0, 0), thickness=-1)

polypts = np.array([
    [650, 100], [520, 175], [520, 325], [650, 400], [780, 325], [780, 175]
])

cv2.polylines(canvas, [polypts], isClosed=True, thickness=4, color=(0, 0, 0))
cv2.circle(canvas, center=(650, 250), radius=130, thickness=-1, color=(0, 0, 0))

cv2.rectangle(canvas, pt1=(900, 450), pt2=(1100, 375), color=(0, 0, 0), thickness=-1)
cv2.rectangle(canvas, pt1=(1025, 375), pt2=(1100, 50), color=(0, 0, 0), thickness=-1)
cv2.rectangle(canvas, pt1=(900, 125), pt2=(1025, 50), color=(0, 0, 0), thickness=-1)

plt.imshow(canvas)
plt.gca().invert_yaxis()
plt.show()