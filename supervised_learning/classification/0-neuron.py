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

        self.W = np.random.normal(size=nx)
        self.b = 0
        self.A = 0
