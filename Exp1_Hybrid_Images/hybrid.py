# coding=utf-8
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import sys

sys.path.append('/Users/kb/bin/opencv-3.1.0/build/lib/')

import numpy as np


def check_odd(kernel):
    """
    检查卷积核的维度是否全为奇数
    """
    for i in range(2):
        # print(kernel.shape)
        if not kernel.shape[i] % 2:
            raise Exception("Both m or n should be odd.")


def zero_pad(img, kernel):
    w_pad = int((kernel.shape[0] - 1) / 2)
    h_pad = int((kernel.shape[1] - 1) / 2)
    img_pad = np.pad(img, ((w_pad, w_pad), (h_pad, h_pad), (0, 0)), str('constant'))

    return img_pad


def gaussian_function(x, y, sigma):
    pi = 3.1415926
    return (1 / (2 * pi * sigma ** 2)) * np.exp(-1 * (x ** 2 + y ** 2) / (2 * sigma ** 2))


def cross_correlation_2d(img, kernel):
    """Given a kernel of arbitrary m x n dimensions, with both m and n being
    odd, compute the cross correlation of the given image with the given
    kernel, such that the output is of the same dimensions as the image and that
    you assume the pixels out of the bounds of the image to be zero. Note that
    you need to apply the kernel to each channel separately, if the given image
    is an RGB image.

    Inputs:
        img:    Either an RGB image (height x width x 3) or a grayscale image
                (height x width) as a numpy array.
        kernel: A 2D numpy array (m x n), with m and n both odd (but may not be
                equal).

    Output:
        Return an image of the same dimensions as the input image (same width,
        height and the number of color channels)
    """

    check_odd(kernel)
    if len(img.shape) == 2:
        img = np.expand_dims(img, axis=-1)

    ch = img.shape[-1]
    result_img = np.zeros(shape=img.shape)
    # 为后续处理方便
    img = zero_pad(img, kernel).transpose((2, 0, 1))

    for k in range(ch):
        for j in range(img.shape[2] - kernel.shape[1] + 1):
            for i in range(img.shape[1] - kernel.shape[0] + 1):
                result_img[i, j][k] = (img[k, i:(i + kernel.shape[0]), j:(j + kernel.shape[1])] * kernel).sum()
    result_img = np.squeeze(result_img)

    return result_img


def convolve_2d(img, kernel):
    """Use cross_correlation_2d() to carry out a 2D convolution.

    Inputs:
        img:    Either an RGB image (height x width x 3) or a grayscale image
                (height x width) as a numpy array.
        kernel: A 2D numpy array (m x n), with m and n both odd (but may not be
                equal).

    Output:
        Return an image of the same dimensions as the input image (same width,
        height and the number of color channels)
    """
    kernel = np.flip(kernel)
    return cross_correlation_2d(img, kernel)


def gaussian_blur_kernel_2d(sigma, height, width):
    """Return a Gaussian blur kernel of the given dimensions and with the given
    sigma. Note that width and height are different.

    Input:
        sigma:  The parameter that controls the radius of the Gaussian blur.
                Note that, in our case, it is a circular Gaussian (symmetric
                across height and width).
        width:  The width of the kernel.
        height: The height of the kernel.

    Output:
        Return a kernel of dimensions height x width such that convolving it
        with an image results in a Gaussian-blurred image.
    """
    kernel = np.zeros(shape=(height, width))

    # 确定中心坐标
    center_x = np.floor(width / 2)
    center_y = np.floor(height / 2)

    for i in range(width):
        for j in range(height):
            kernel[j, i] = gaussian_function(i - center_x, j - center_y, sigma)
    # 标准化
    kernel = kernel / kernel.sum()

    return kernel


def low_pass(img, sigma, size):
    """Filter the image as if its filtered with a low pass filter of the given
    sigma and a square kernel of the given size. A low pass filter supresses
    the higher frequency components (finer details) of the image.

    Output:
        Return an image of the same dimensions as the input image (same width,
        height and the number of color channels)
    """
    kernel = gaussian_blur_kernel_2d(sigma, size, size)
    img = convolve_2d(img, kernel)
    return img


def high_pass(img, sigma, size):
    """Filter the image as if its filtered with a high pass filter of the given
    sigma and a square kernel of the given size. A high pass filter suppresses
    the lower frequency components (coarse details) of the image.

    Output:
        Return an image of the same dimensions as the input image (same width,
        height and the number of color channels)
    """
    kernel = gaussian_blur_kernel_2d(sigma, size, size)
    # print(g)
    img = img - convolve_2d(img, kernel)
    return img


def create_hybrid_image(img1, img2, sigma1, size1, high_low1, sigma2, size2,
                        high_low2, mixin_ratio):
    """This function adds two images to create a hybrid image, based on
    parameters specified by the user."""
    high_low1 = high_low1.lower()
    high_low2 = high_low2.lower()

    if img1.dtype == np.uint8:
        img1 = img1.astype(np.float32) / 255.0
        img2 = img2.astype(np.float32) / 255.0

    if high_low1 == 'low':
        img1 = low_pass(img1, sigma1, size1)
    else:
        img1 = high_pass(img1, sigma1, size1)

    if high_low2 == 'low':
        img2 = low_pass(img2, sigma2, size2)
    else:
        img2 = high_pass(img2, sigma2, size2)

    img1 *= 2 * (1 - mixin_ratio)
    img2 *= 2 * mixin_ratio
    hybrid_img = (img1 + img2)
    return (hybrid_img * 255).clip(0, 255).astype(np.uint8)
