#!/usr/bin/env python3
'''6-multihead_attention'''


sdp_attention = __import__('5-sdp_attention').sdp_attention
import tensorflow as tf


class MultiHeadAttention:
    '''scsmncmjjmkjdm
    nnkbknjgkmc'''
    def __init__(self, dm, h):
        '''initializer'''
        self.h = h
        self.dm = dm
        self.depth = dm // h
        self.Wq = tf.keras.layers.Dense(dm)
        self.Wk = tf.keras.layers.Dense(dm)
        self.Wv = tf.keras.layers.Dense(dm)
        self.linear = tf.keras.layers.Dense(dm)

    def __call__(self, Q, K, V, mask):
        '''call function'''
        Q = self.Wq(Q)
        K = self.Wk(K)
        V = self.Wv(V)

        Q = tf.transpose(tf.reshape(Q, (Q.shape[0], Q.shape[1], self.h,
                                        self.depth)), perm=[0, 2, 1, 3])
        K = tf.transpose(tf.reshape(K, (K.shape[0], K.shape[1], self.h,
                                        self.depth)), perm=[0, 2, 1, 3])
        V = tf.transpose(tf.reshape(V, (V.shape[0], V.shape[1], self.h,
                                        self.depth)), perm=[0, 2, 1, 3])

        output, weights = sdp_attention(Q, K, V)
        output = tf.identify(tf.transpose(output, perm=[0, 2, 1, 3]))
        output = tf.reshape(output, [output.shape[0], output.shape[1], -1])

        return output, weights
