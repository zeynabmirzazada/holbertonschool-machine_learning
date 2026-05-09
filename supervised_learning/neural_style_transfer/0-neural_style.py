#!/usr/bin/env python3
'''initialize'''
import numpy as np
import tensorflow as tf


class NST():
    '''init'''
    style_layers = ['block1_conv1', 'block2_conv1', 'block3_conv1',
                    'block4_conv1', 'block5_conv1']
    content_layer = 'block5_conv2'
    def __init__(self, style_image, content_image, alpha=1e4, beta=1):
        '''instance'''
        if len(style_image.shape) != 3 or style_image.shape[2] != 3:
            raise TypeError('style_image must be a numpy.ndarray with shape'
            '(h, w, 3)')
        if len(content_image.shape) != 3 or content_image.shape[2] != 3:
            raise TypeError('content_image must be a numpy.ndarray with ' \
            'shape (h, w, 3)')
        self.style_image = style_image
        self.content_image = content_image
        self.alpha = alpha
        self.beta = beta
    @staticmethod
    def scale_image(image):
        '''scaling'''
        if len(image.shape) != 3 or image.shape[2] != 3:
            raise TypeError('image must be a numpy.ndarray with shape'
            '(h, w, 3)')
        
        h, w = image.shape[0], image.shape[1]
        sc = 512 / max(h, w)
        new_h, new_w = int(sc * h), int(sc * w)
        #img = tf.convert_to_tensor(image, dtype=tf.float32)
        img = tf.image.resize(image, [new_h, new_w], method='bicubic')
        img = tf.expand_dims(img, axis=0)
        return tf.cast(img, tf.float32) / 255.0
