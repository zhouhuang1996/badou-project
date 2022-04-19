import cv2
import numpy as np


def get_gray_from_bgr(source_file):

    img = cv2.imread(source_file)
    img_grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    return img_grey


def bgr_to_gray(source_file, target_file):

    img_grey = get_gray_from_bgr(source_file)
    cv2.imwrite(target_file, img_grey)


def bgr_to_blank_and_white(source_file, target_file):

    img_grey = get_gray_from_bgr(source_file)
    img_black_white = np.zeros(img_grey.shape, img_grey.dtype)

    rows = img_grey.shape[0]
    columns = img_grey.shape[1]

    for i in range(rows):
        for j in range(columns):
            if img_grey[i, j] / 255 >= 0.5:
                img_black_white[i, j] = 255
            else:
                img_black_white[i, j] = 0

    cv2.imwrite(target_file, img_black_white)