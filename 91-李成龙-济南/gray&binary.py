import numpy as np
import matplotlib.pyplot as plt
from skimage.color import rgb2gray
from PIL import Image

image = plt.imread("lena.jpg")
plt.subplot(221)
plt.imshow(image)
# plt.show()
[h, w] = image.shape[:2]
# #########实现灰度化###########
image_gray = np.zeros([h, w], np.uint8)
for i in range(h):
    for j in range(w):
        img = image[i, j]
        image_gray[i, j]= img[0]*0.3+img[1]*0.59+img[2]*0.11  # 将RGB值转化为灰度值，并赋给新的图像空间。(浮点算法)
        # image_gray[i, j] = (img[0] * 11 + img[1] * 59 + img[2] * 30) / 100 #（整数方法）
        # image_gray[i, j] = (img[0] + img[1] + img[2]) / 3 # 取均值法
        # image_gray[i, j] = img[2]  # 仅取一种颜色
print("image_gray的灰度值:", image_gray)
plt.subplot(222)
plt.imshow(image_gray, cmap="gray")
# plt.imshow(image_gray)
# plt.show()


# ##########实现二值化###########
image_gray = rgb2gray(image)
m, n = image_gray.shape[:2]
# print(image_gray.shape[:2])
image_binary = np.zeros([m, n], np.uint8)
for i in range(m):
    for j in range(n):
        if image_gray[i, j] <= 0.5:
            image_gray[i, j] = 0
            image_binary[i, j] = image_gray[i, j]
        else:
            image_gray[i, j] = 1
            image_binary[i, j] = image_gray[i, j]
print("image_binary二值化的值为", image_binary)
plt.subplot(223)
plt.imshow(image_binary)
plt.show()
