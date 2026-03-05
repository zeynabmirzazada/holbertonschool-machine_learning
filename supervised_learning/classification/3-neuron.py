#!/usr/bin/env python3
"""Module that defines a single neuron performing binary classification."""

import numpy as np


class Neuron:
    """Neuron class that defines a single neuron for binary classification."""

    def __init__(self, nx):
        """
        Initialize a Neuron instance.

        Parameters:
        nx (int): Number of input features.

        Raises:
        TypeError: If nx is not an integer.
        ValueError: If nx is less than 1.
        """
        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")

        self.__W = np.random.normal(size=(1, nx))
        self.__b = 0
        self.__A = 0

    def forward_prop(self, X):
        """calculate forward propagation"""
        s = np.matmul(self.__W, X) + self.__b
        self.__A = 1 / (1 + np.e ** (-s))
        return self.__A

    def cost(self, Y, A):
        '''logistic cost function'''
        return -np.mean(Y * np.log(A) + (1 - Y) * np.log(1.0000001 - A))

    @property
    def W(self):
        return self.__W

    @property
    def b(self):
        return self.__b

    @property
    def A(self):
        return self.__A
