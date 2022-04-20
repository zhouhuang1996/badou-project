# -*- coding:utf-8 -*-
# 八斗AI课作业.
# 图像处理工具类
# 42-吴清-武汉
# 2022-4-17

import numpy as np
import cv2 as cv

# 灰度化函数
# _filePath 图像文件路径
def image_gray(_filePath):
    img = cv.imread(_filePath)
    print("img:", img)
    h, w = img.shape[:2]
    img_gray = np.zeros([h, w], img.dtype)  # 创建一张指定长宽的空白图
    print("img width:%d height:%d" % (h, w))
    print("img point 0,0 channel: %d" % (len(img[0, 0])))
    for i in range(h):
        for j in range(w):
            m = img[i, j]  # 获取源图i,j的像素图
            img_gray[i, j] = int(m[0] * 0.11 + m[1] * 0.59 + m[2] * 0.3)  # 给灰度图i，j点像素赋值
    print("img_gray point 0,0 ", img_gray[0, 0])
    print("img_gray:", img_gray)
    cv.imshow("gray image", img_gray)


# 二值化图片
# _filePath 图像文件路径
def image_twoValue(_filePath):
    img = cv.imread(_filePath)
    print("img:", img)
    h, w = img.shape[:2]
    img_gray = np.zeros([h, w], img.dtype)  # 创建一张指定长宽的空白图

    for i in range(h):
        for j in range(w):
            m = img[i, j]  # 获取源图i,j的像素图
            img_gray[i, j] = 0 if int(m[0] * 0.11 + m[1] * 0.59 + m[2] * 0.3) < 125 else 255  # 给二值图i，j点像素赋值
    print("img_gray:", img_gray)
    cv.imshow("show image", img_gray)