import cv2 as cv
import numpy as np

## 1. 灰度化
img = cv.imread("lena.png")
cv.imshow("lena", img)

img_gray1 = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
cv.imshow("lena_gray1", img_gray1)

height = img.shape[0]
width = img.shape[1]
img_gray2 = np.zeros([height, width], img.dtype)                   # 创建一张和当前图片大小一样的单通道图片
for i in range(height):
    for j in range(width):
        m = img[i, j]                                               # 取出当前high和wide中的BGR坐标
        img_gray2[i, j] = int(m[0]*0.11 + m[1]*0.59 + m[2]*0.3)     # 将BGR坐标转化为gray坐标并赋值给新图像
print (img_gray2)
cv.imshow("lena_gray2", img_gray2)

err_img = img_gray1 - img_gray2
print("err_img is: ")
print(err_img)


## 2. 图像二值化
ret, img_binary1 = cv.threshold(img_gray1, 126, 255, cv.THRESH_BINARY)
cv.imshow("lena_thresh_1", img_binary1)

img_binary2 = np.where(img_gray1 >= 126, 255, 0)
img_binary2 = np.array(img_binary2, dtype=np.uint8)  # show出来的图片的类型是：int32，因此要做类型转换
cv.imshow("lena_thresh_2", img_binary2)

err_img = img_binary1 - img_binary2
print("err_img is: ")
print(err_img)

cv.waitKey(0)              # 等待键盘按键
cv.destroyAllWindows()     # 销毁所有窗口