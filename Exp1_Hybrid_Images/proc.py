# coding=utf-8
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import numpy as np
import cv2
from hybrid import *


def fun1():
    img1 = cv2.imread('cat.jpg')
    img2 = cv2.imread('dog.jpg')
    size = 15
    sigma = 5
    # kernel = gaussian_blur_kernel_2d(15, 55, 55)
    output = high_pass(img1, sigma, size) + low_pass(img2, sigma, size)
    cv2.imwrite('hybrid_923.png', output)


def fun2():
    names = ['cat_dog2', 'sad_happy2', 'cycle_motorcycle2']

    times = [2, 4, 8, 16]

    for name in names:
        for time in times:
            img = cv2.imread(name + '.png')
            x, y = img.shape[0:2]
            img_resize = cv2.resize(img, (int(y / time), int(x / time)))
            cv2.imwrite(name + '_' + str(time) + '.png', img_resize)


def fun3():
    img1 = cv2.imread('cat.jpg')
    img2 = cv2.imread('dog.jpg')
    sigma = 5
    for size in [5, 15, 25, 35]:
        # kernel = gaussian_blur_kernel_2d(15, 55, 55)
        high = high_pass(img1, sigma, size)
        low = low_pass(img2, sigma, size)
        cv2.imwrite('high_size_' + str(size) + '.png', high)
        cv2.imwrite('low_size_' + str(size) + '.png', low)
        cv2.imwrite('hybrid_size_' + str(size) + '.png', high+low)


if __name__ == '__main__':
    fun3()
