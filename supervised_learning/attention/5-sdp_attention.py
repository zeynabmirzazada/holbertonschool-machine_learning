#!/usr/bin/env python3
'''sdccvgkjsdccfgffgk'''


import tensorflow as tf


def sdp_attention(Q, K, V, mask=None):
    '''scldglmflkkgsmncmjjmkjdm'''
    a = (Q @ tf.transpose(K, perm=[0, 2, 1])) / (
        tf.math.sqrt(tf.constant(float(K.shape[-1]))))
    if mask != None:
        a += tf.math.multiply(mask, -1e9)

    weights = tf.exp(a) / tf.reduce_sum(a, axis=-1, keepdims=True)

    output = tf.matmul(weights, V)
    return output, weights
