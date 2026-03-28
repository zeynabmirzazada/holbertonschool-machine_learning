#!/usr/bin/env python3
'''strided convolution'''
import numpy as np


def convolve_grayscale(images, kernel, padding='same', stride=(1, 1)):
    '''images is a numpy.ndarray with shape (m, h, w) containing multiple
    grayscale images
        m is the number of images
        h is the height in pixels of the images
        w is the width in pixels of the images
    kernel is a numpy.ndarray with shape (kh, kw) containing the kernel for
    convolution
        kh is the height of the kernel
        kw is the width of the kernel
    padding is either a tuple of (ph, pw), 'same', or 'valid'
        if 'same', performs a same convolution
        if 'valid', performs a valid convolution
        if a tuple:
            ph is the padding for the height of the image
            pw is the padding for the width of the image
        the image should be padded with 0's
    stride is a tuple of (sh, sw)
        sh is the stride for the height of the image
        sw is the stride for the width of the image
    You are only allowed to use two for loops; Hint: loop over i and j
    Returns: a numpy.ndarray containing the convolved images'''
    m, h, w = images.shape[0], images.shape[1], images.shape[2]
    kh, kw = kernel.shape[0], kernel.shape[1]
    sh, sw = stride[0], stride[1]
    if padding == 'same':
        ph, pw = kh // 2, kw // 2
    if padding == 'valid':
        ph, pw = 0, 0
    if isinstance(padding, tuple):
        ph, pw = padding[0], padding[1]
    images = np.pad(images,
                    ((0, 0), (ph, ph),
                        (pw, pw)),
                    constant_values=0)
    conv = np.zeros((m, (images.shape[1] - kh) // sh + 1,
                    (images.shape[2] - kw) // sw + 1))
    for row in range(0, images.shape[1] - kh + 1, sh):
        for column in range(0, images.shape[2] - kw + 1, sw):
            part = images[:, row:row + kh, column:column + kw] * kernel
            conv[:, (row // 2), (column // 2)] = np.sum(part, axis=(1, 2))
    return conv
