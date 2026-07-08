#!/usr/bin/env python3
'''dghth'''
import tensorflow as tf


class RNNEncoder:
    '''dvb'''
    def __init__(self, vocab, embedding, units, batch):
        '''vcbgfg'''
        self.batch = batch
        self.units = units
        self.embedding = tf.keras.layers.Embedding(vocab, embedding)
        self.gru = tf.keras.layers.GRU(units,
                                       recurrent_initializer='glorot_uniform',
                                       return_sequences=True,
                                       return_state=True)

    def initialize_hidden_state(self):
        '''dhfhg'''
        return tf.zeros([self.batch, self.units])
    
    def __call__(self, x, initial):
        '''cnnncg'''
        a = self.gru(self.embedding(x), initial_state=initial)
        return a
