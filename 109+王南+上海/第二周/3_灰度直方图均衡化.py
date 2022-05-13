import numpy as np
import cv2
import matplotlib.pyplot as plt


image = cv2.imread("lena.jpeg")
image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
image_hist = cv2.calcHist([image_gray], [0], None, [256], [0, 255])
image_gray_eq = cv2.equalizeHist(image_gray)
image_hist_eq = cv2.calcHist([image_gray_eq], [0], None, [256], [0, 255])

plt.subplot(2, 2, 1)
plt.axis("off")
plt.imshow(cv2.cvtColor(image_gray, cv2.COLOR_BGR2RGB))

plt.subplot(2, 2, 2)
plt.xlabel("bins")
plt.ylabel("pixels")
plt.xlim([0, 256])
plt.plot(image_hist, color="m")

plt.subplot(2, 2, 3)
plt.axis("off")
plt.imshow(cv2.cvtColor(image_gray_eq, cv2.COLOR_BGR2RGB))

plt.subplot(2, 2, 4)
plt.xlabel("bins")
plt.ylabel("pixels")
plt.xlim([0, 256])
plt.plot(image_hist_eq, color="m")

plt.show()