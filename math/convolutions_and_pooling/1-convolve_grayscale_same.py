#!/usr/bin/env python3
'''same convolution on grayscale images'''
import numpy as np


def convolve_grayscale_same(images, kernel):
    '''
    images is a numpy.ndarray with shape (m, h, w) containing multiple
    grayscale images
        m is the number of images
        h is the height in pixels of the images
        w is the width in pixels of the images
    kernel is a numpy.ndarray with shape (kh, kw) containing the kernel for th
    convolution
        kh is the height of the kernel
        kw is the width of the kernel
    Returns: a numpy.ndarray containing the convolved images'''
    m, h, w = images.shape[0], images.shape[1], images.shape[2]
    kh, kw = kernel.shape[0], kernel.shape[1]
    conv_imgs = np.zeros((m, h, w))
    for row in range(h - kh + 1):
        for column in range(w - kw + 1):
            part = images[:, row:row + kh, column:column + kw] * kernel
            conv_imgs[:, row + 1, column + 1] = np.sum(part, axis=(1, 2))
    return conv_imgs
