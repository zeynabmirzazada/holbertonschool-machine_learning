#!/usr/bin/env python3
'''build model'''
import tensorflow.keras as K


def build_model(nx, layers, activations, lambtha, keep_prob):
    '''nx is the number of input features to the network
    layers - is a list containing the number of nodes in each layer of the
    network
    activations - is a list containing the activation functions used for each
    layer of the network
    lambtha - is the L2 regularization parameter
    keep_prob is the probability that a node will be kept for dropout
    You are not allowed to use the Input class
    Returns: the keras model'''
    model = K.Sequential([
        K.layers.Dense(layers[0],
                       kernel_regularizer=K.regularizers.L2(lambtha),
                       input_shape=(nx,), activation=activations[0]),
        K.layers.Dropout(keep_prob),  # Dropout layer with 30% rate
        K.layers.Dense(layers[1],
                       kernel_regularizer=K.regularizers.L2(lambtha),
                       activation=activations[1]),
        K.layers.Dropout(keep_prob),
        K.layers.Dense(layers[2],
                       kernel_regularizer=K.regularizers.L2(lambtha),
                       activation=activations[2])
        ])
    return model
