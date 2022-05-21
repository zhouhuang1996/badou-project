import cv2
import matplotlib.pyplot as plt

image = cv2.imread("lena.jpeg")
h, s, v = cv2.split(cv2.cvtColor(image, cv2.COLOR_BGR2HSV))

v_eq = cv2.equalizeHist(v)
image_eq = cv2.cvtColor(cv2.merge([h, s, v_eq]), cv2.COLOR_HSV2RGB)

image_hist = cv2.calcHist(v, [0], None, [256], [0, 255])
image_eq_hist = cv2.calcHist(v_eq, [0], None, [256], [0, 255])

# 需要matplotlib中安装中文字体
# plt.rcParams["font.sans-serif"] = ["SimHei"]
# plt.rcParams["axes.unicode_minus"] = False

plt.subplot(2, 2, 1)
plt.title("source image")
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.axis("off")

plt.subplot(2, 2, 2)
plt.xlabel("bins")
plt.ylabel("pixels")
plt.xlim([0, 255])
plt.plot(image_hist, color="m")

plt.subplot(2, 2, 3)
plt.title("target image")
plt.imshow(image_eq)
plt.axis("off")

plt.subplot(2, 2, 4)
plt.xlabel("bins")
plt.ylabel("pixels")
plt.xlim([0, 255])
plt.plot(image_eq_hist, color="m")

plt.show()