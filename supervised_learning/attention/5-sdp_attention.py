#!/usr/bin/env python3
'''sdccvgkjsdcck'''


import tensorflow as tf


def sdp_attention(Q, K, V, mask=None):
    '''sc smncmjjmkjdm'''
    a = tf.matmul(Q, K, transpose_b=True) /
        tf.math.sqrt(tf.constant(float(K.shape[-1])))
    if mask != None:
        a += mask * -1e9

    weights = tf.nn.softmax(a, axis=-1)

    output = tf.matmul(weights, V)
    return output, weights
