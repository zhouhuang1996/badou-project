import cv2

img = cv2.imread("lenna.png")
# cv2.imshow('', img)
imggray = img
imggray[:,:,0] = img[:,:,0]
imggray[:,:,1] = img[:,:,0]
imggray[:,:,2] = img[:,:,0]
cv2.imshow('',imggray)
cv2.waitKey(0)

m1, n1 = imggray.shape[0:2]
for m in range(m1):
    for n in range(n1):
        a = imggray[m, n, 0]
        if a < 128:
            a = 255
        else:
            a = 0
        imggray[m, n, 0] = a
        imggray[m, n, 1] = a
        imggray[m, n, 2] = a

cv2.imshow('',imggray)
cv2.waitKey(0)
