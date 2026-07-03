#!/usr/bin/env python3
'''gensim to keras'''
from tensorflow.keras.layers import Embedding


def gensim_to_keras(model):
    '''hn'''
    keyed_vectors = model.wv  # structure holding the result of training
    weights = keyed_vectors.vectors  # vectors themselves, a 2D numpy array
    index_to_key = keyed_vectors.index_to_key  # which row in `weights` corresponds to which word?

    layer = Embedding(
        input_dim=weights.shape[0],
        output_dim=weights.shape[1],
        weights=[weights],
        trainable=False,
    )
    return layer
