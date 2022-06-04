import cv2
import matplotlib.pyplot as plt

image = cv2.imread("lena.jpeg")
channels = cv2.split(image)
channels_hist = []
channels_eq = []
channels_eq_hist = []

for ch in channels:
    ch_eq = cv2.equalizeHist(ch)
    channels_eq.append(ch_eq)
    channels_hist.append(cv2.calcHist(ch, [0], None, [256], [0, 255]))
    channels_eq_hist.append(cv2.calcHist(ch_eq, [0], None, [256], [0, 255]))

image_eq = cv2.merge(channels_eq)

plt.subplot(2, 2, 1)
plt.axis("off")
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

plt.subplot(2, 2, 2)
plt.xlabel("bins")
plt.ylabel("pixels")
plt.xlim([0, 255])
for ch, color in zip(channels_hist, ["b", "g", "r"]):
    plt.plot(ch, color=color)

plt.subplot(2, 2, 3)
plt.axis("off")
plt.imshow(cv2.cvtColor(image_eq, cv2.COLOR_BGR2RGB))

plt.subplot(2, 2, 4)
plt.xlabel("bins")
plt.ylabel("pixels")
plt.xlim([0, 255])
for ch, color in zip(channels_eq_hist, ["b", "g", "r"]):
    plt.plot(ch, color=color)

plt.show()