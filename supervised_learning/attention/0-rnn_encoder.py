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
        return torch.zeros(self.batch, self.units)
    
    def call(self, x, initial):
        '''cnnncg'''
        return self.gru(self.embedding(x))
