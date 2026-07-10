#!/usr/bin/env python3
sdp_attention = __import__('5-sdp_attention').sdp_attention
'''6-multihead_attention'''
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
        mask = None
        Qo = self.Wq(Q)
        Ko = self.Wk(K)
        Vo = self.Wv(V)

        Qo = tf.transpose(tf.reshape(Qo, (Qo.shape[0], Qo.shape[1], self.h,
                                          self.depth)), perm=[0, 2, 1, 3])
        Ko = tf.transpose(tf.reshape(Ko, (Ko.shape[0], Ko.shape[1], self.h,
                                          self.depth)), perm=[0, 2, 1, 3])
        Vo = tf.transpose(tf.reshape(Vo, (Vo.shape[0], Vo.shape[1], self.h,
                                          self.depth)), perm=[0, 2, 1, 3])

        output, weights = sdp_attention(Qo, Ko, Vo)
        output = tf.identity(tf.transpose(output, perm=[0, 2, 1, 3]))
        output = tf.reshape(output, [output.shape[0], output.shape[1], -1])

        return self.linear(output), weights
