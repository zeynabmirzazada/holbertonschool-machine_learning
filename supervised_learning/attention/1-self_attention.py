#!/usr/bin/env python3
'''Self Attention'''


import tensorflow as tf


class SelfAttention:
    '''class of self attention'''
    def __init__(self, units):
        '''initialize'''
        self.W = tf.keras.layers.Dense(units)
        self.U = tf.keras.layers.Dense(units)
        self.V = tf.keras.layers.Dense(1, activation='tanh')

    def __call__(self, s_prev, hidden_states):
        '''dbdfbdfxv'''
        w = self.W(s_prev)
        u = self.U(hidden_states)
        v = self.V(tf.math.tanh(tf.expand_dims(w, axis=1) + u))
        return w, v
