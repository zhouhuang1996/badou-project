import numpy as np
import cv2


# 最邻近插值缩放图片
def nearest_translate(img, shape):
    h, w, c = img.shape
    h_rate = shape[0] / h
    w_rate = shape[1] / w
    new_img = np.zeros((shape[0], shape[1], c), dtype=img.dtype)
    for i in range(shape[0]):
        for j in range(shape[1]):
            x = int(i / h_rate + 0.5)
            y = int(j / w_rate + 0.5)
            new_img[i, j] = img[x, y]
    return new_img


eagle_img = cv2.imread("eagle.jpeg")
print(eagle_img.shape)
cv2.imshow("eagle", eagle_img)
# 放大图片
cv2.imshow("eagle_1000_2000", nearest_translate(eagle_img, (1000, 2000)))
# 缩小图片
cv2.imshow("eagle_600_800", nearest_translate(eagle_img, (600, 800)))

while cv2.waitKey(0) != 32:
    pass
