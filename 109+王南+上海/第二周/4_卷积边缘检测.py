import cv2
import numpy as np

image = cv2.cvtColor(cv2.imread("eagle.jpeg"), cv2.COLOR_BGR2GRAY)
print(image.shape)
h, w = image.shape
# 图像矩阵四周加一圈0
image_padding = np.zeros((h + 2, w + 2), dtype=image.dtype)
print(image_padding.shape)
image_padding[1:h+1, 1:w+1] = image
h_padding, w_padding = image_padding.shape
image_out = np.zeros(image_padding.shape, dtype=image.dtype)

kernel = [
    [1, 0, -1],
    [1, 0, -1],
    [1, 0, -1]
]
#
# temp = np.array([
#     [1, 2, 3],
#     [4, 5, 6],
#     [7, 8, 9],
# ])
#
# print(kernel * temp)
# print(np.sum(kernel * temp))
# print(np.dot(kernel, temp))
# print(np.dot(temp, kernel))
# print(np.matmul(kernel, temp))
# print(np.matmul(temp, kernel))
# print(image[1:4, 1:4])
# print(image[0:6, 0:6])


for j in range(1, h_padding - 1):
    for i in range(1, w_padding - 1):
        image_out[j, i] = np.sum(kernel * image_padding[j-1:j+2, i-1:i+2])

cv2.imshow("卷积边缘检测", image_out)
while cv2.waitKey(0) != 32:
    pass
cv2.destroyAllWindows()
exit(0)