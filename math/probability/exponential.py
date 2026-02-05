#!/usr/bin/env python3
"""
Contains the Exponential class which represents an exponential distribution.
"""


class Exponential:
    """
    Class that represents an exponential distribution.
    """

    def __init__(self, data=None, lambtha=1.0):
        """
        Initializes the Exponential distribution.

        Args:
            data (list): A list of data used to estimate the distribution.
            lambtha (float): The expected number of occurrences.

        Raises:
            TypeError: If data is not a list.
            ValueError: If lambtha is not positive.
            ValueError: If data contains fewer than two data points.
        """
        if data is None:
            if lambtha <= 0:
                raise ValueError("lambtha must be a positive value")
            self.lambtha = float(lambtha)
        else:
            if not isinstance(data, list):
                raise TypeError("data must be a list")
            if len(data) < 2:
                raise ValueError("data must contain multiple values")
            
            # For exponential distribution, lambtha = 1 / mean
            # Mean = sum(data) / len(data)
            self.lambtha = float(len(data) / sum(data))
