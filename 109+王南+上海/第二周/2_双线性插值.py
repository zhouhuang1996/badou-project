import numpy as np
import cv2


def bi_linear_translate(src, shape):
    # 源图像高、宽、通道数
    h_src, w_src, channels = src.shape
    # 目标图像高、宽
    h_dest, w_dest = shape
    # 源图像与目标图像高、宽比例
    h_rate = h_src / h_dest
    w_rate = w_src / w_dest
    dest = np.zeros((h_dest, w_dest, channels), dtype=src.dtype)

    for c in range(channels):
        for y_dest in range(h_dest):
            for x_dest in range(w_dest):
                # 根据目标图像像素坐标计算映射到源图像的像素坐标
                # （坐标为浮点数，在源图像上没有对应真实的像素坐标）
                # 加减0.5是处理源图和目标图几何中心重合
                y_src = (y_dest + 0.5) * h_rate - 0.5
                x_src = (x_dest + 0.5) * w_rate - 0.5
                # 根据源图像像素坐标计算临近四个真实像素坐标
                # (x1, y1), (x1, y2), (x2, y1), (x2, y2)
                x1 = int(np.floor(x_src))
                x2 = min(x1 + 1, w_src - 1)
                y1 = int(np.floor(y_src))
                y2 = min(y1 + 1, h_src - 1)
                # 计算中间像素值R1,R2。 x2 - x1 = 1 所以省略分母
                # f(R1) = f(x, y1) = (x2 - x)/(x2 - x1) * f(x1, y1) + (x - x1)/(x2 - x1) * f(x2, y1)
                # f(R2) = f(x, y2) = (x2 - x)/(x2 - x1) * f(x1, y2) + (x - x1)/(x2 - x1) * f(x2, y2)
                f_r1 = (x2 - x_src) * src[y1, x1, c] + (x_src - x1) * src[y1, x2, c]
                f_r2 = (x2 - x_src) * src[y2, x1, c] + (x_src - x1) * src[y2, x2, c]
                # 计算目标像素值 y2 - y1 = 1 所以省略分母
                # f(x, y) = (y2 - y) / (y2 - y1) * f(R1) + (y - y1) / (y2 - y1) * f(R2)
                dest[y_dest, x_dest, c] = (y2 - y_src) * f_r1 + (y_src - y1) * f_r2
    return dest


img = cv2.imread("eagle.jpeg")
cv2.imshow("eagle", img)
# cv2.imshow("eagle_1000_2000", bi_linear_translate(img, (1000, 2000)))
# cv2.imshow("eagle_600_800", bi_linear_translate(img, (600, 800)))
cv2.imshow("eagle_200_300", bi_linear_translate(img, (200, 300)))
cv2.imshow("eagle_50_100", bi_linear_translate(img, (50, 100)))

while cv2.waitKey(0) != 32:
    pass

cv2.destroyAllWindows()
exit(0)
