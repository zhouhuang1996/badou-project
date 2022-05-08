# -*- coding:utf-8 -*-
# 八斗AI课作业.
# 图像处理工具类
# 42-吴清-武汉
# 2022-4-17
#原始lenna图尺寸372*374

import numpy as np
import cv2 as cv

# 灰度化函数
# _filePath 图像文件路径
# 2022-4-17
def image_gray(_filePath):
    img = cv.imread(_filePath)
    print("source img:", img)
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
    return img_gray

# 二值化图片
# _filePath 图像文件路径
# 2022-4-17
def image_twoValue(_filePath):
    img = cv.imread(_filePath)
    print("source img:", img)
    h, w = img.shape[:2]
    img_gray = np.zeros([h, w], img.dtype)  # 创建一张指定长宽的空白图

    for i in range(h):
        for j in range(w):
            m = img[i, j]  # 获取源图i,j的像素图
            img_gray[i, j] = 0 if int(m[0] * 0.11 + m[1] * 0.59 + m[2] * 0.3) < 125 else 255  # 给二值图i，j点像素赋值
    print("img_gray:", img_gray)
    return img_gray

# 邻近插值算法
# _filePath 图像文件路径
# x_size 新的x尺寸
# y_size 新的y尺寸
# 2022-5-4
def image_adjacent(_filePath,x_size,y_size):
    img = cv.imread(_filePath)
    print("source img:", img)
    s_x, s_y,s_channels = img.shape # 获取原始图像的尺寸
    print("source img size:",s_x,s_y)
    img_target = np.zeros((x_size, y_size,s_channels), img.dtype)  # 创建一张新的指定长宽的空白图
    for i in range(x_size):
        for j in range(y_size):
            source_point_x=int(i*s_x/x_size) # 获取在原始图中的邻近点x坐标
            source_point_y = int(j*s_y/y_size) # 获取在原始图中的邻近点y坐标
            # print("current source point :",source_point_x,source_point_y,img[source_point_x,source_point_y])
            img_target[i,j]=img[source_point_x,source_point_y]
    print ("img_adjacent",img_target)
    return img_target

# 双线性插值算法
# _filePath 图像文件路径
# target_x 新的x尺寸
# target_y 新的y尺寸
# 2022-5-4
def image_linear(_filePath,target_x,target_y):
    img = cv.imread(_filePath)
    print("source img:", img)
    s_x, s_y,s_channels = img.shape # 获取原始图像的尺寸
    print("source img size:",s_x,s_y)
    img_target = np.zeros((target_x, target_y,s_channels), img.dtype)  # 创建一张新的指定长宽的空白图
    for c in range(3):
        for i in range(target_x):
            for j in range(target_y):
                #获取当前点在经过中心重合对齐以及缩放后在原图的位置（P点）
                point_p_x=( i+ 0.5) * s_x/target_x-0.5
                #y=(x1-x)y0+(x-x0)y1
                point_p_y = (j + 0.5) * s_y/target_y-0.5
                #获取P点在原图中邻近的4个点x0,x1,y0,y1
                point_s_x0=int(max(point_p_x,0)) #防止起点为负数
                point_s_x1=int(min(point_s_x0 + 1 ,s_x - 1)) # 防止x+1点在原图中不存在
                point_s_y0=int(max(point_p_y,0)) #防止起点为负数
                point_s_y1=int(min(point_s_y0 + 1, s_y - 1)) #防止y+1点在原图中不存在
                #根据双线性插值公式，获取当前点的当前通道值
                #f(L1)=(x1-x)f(0,0)+(x-x0)f(1,0)
                #f(L2)=(x1-x)f(0,1)+(x-x0)f(1,1)
                #f(P)=(y1-y)f(L1)+(y-y1)f(L2)
                L1= (point_s_x1 - point_p_x)*img[point_s_x0, point_s_y0, c]+ (point_p_x - point_s_x0) * img[point_s_x1, point_s_y0, c]
                L2= (point_s_x1 - point_p_x)*img[point_s_x0, point_s_y1, c]+ (point_p_x - point_s_x0) * img[point_s_x1, point_s_y1, c]
                img_target[i, j, c] = int((point_s_y1 - point_p_y) * L1 + (point_p_y - point_s_y0) * L2)
                #img_target[i, j, c] = int(img[point_s_x0,point_s_y0,c])

    print ("img_adjacent",img_target)
    return img_target

# 直方图均衡化
# _filePath 图像文件路径
# 2022-05-04
def image_histogram(_filePath):
    img_gray=image_gray(_filePath)
    h,w = img_gray.shape[:2]
    img_histogram = np.zeros([h, w], img_gray.dtype)  # 创建一张新的指定长宽的空白图
    histogramArray=[0]*256
    LUT=[0]*256
    for i in range(h):
        for j in range(w):
            grayValue=img_gray[i, j] #当前点的灰度值
            histogramArray[grayValue]=histogramArray[grayValue]+1 #记录当前灰度值的数量
    #计算累积分布直方图
    h_sum=histogramArray[0]
    for i in range(256):
        h_sum+=histogramArray[i]
        LUT[i]=255* h_sum / (w*h) #存储均衡化后的值
    for i in range(h):
        for j in range(w):
            # 均衡化处理
            img_histogram[i, j] = LUT[img_gray[i,j]]
    print("img_histogram:", img_histogram)
    return img_histogram
