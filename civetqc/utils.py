import cv2 as cv
import matplotlib.pyplot as plt


def disp(img):
    plt.figure(figsize = (30, 10), dpi = 150)
    if img.ndim == 2:
        plt.imshow(img)
    elif img.ndim == 3:
        plt.imshow(cv.cvtColor(img, cv.COLOR_BGR2RGB))