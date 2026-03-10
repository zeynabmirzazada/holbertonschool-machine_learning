#!/usr/bin/env python3
'''build model'''
import keras


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
    model = Sequential([
        Dense(layers[0], kernel_regularizer=regularizers.L2(lambtha),
            input_shape=(nx,), activation=activations[0]),
        Dropout(keep_prob),  # Dropout layer with 30% rate
        Dense(layers[1], kernel_regularizer=regularizers.L2(lambtha),
            activation=activations[1]),
        Dropout(keep_prob),
        Dense(layers[2], kernel_regularizer=regularizers.L2(lambtha),
            activation=activations[2])
])
