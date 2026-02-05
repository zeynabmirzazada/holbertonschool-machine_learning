#!/usr/bin/env python3
"""
Contains the Poisson class which represents a Poisson distribution.
"""


class Poisson:
    """
    Class that represents a Poisson distribution.
    """

    def __init__(self, data=None, lambtha=1.0):
        """
        Initializes the Poisson distribution.

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
            # Calculate lambtha as the mean of the data
            self.lambtha = float(sum(data) / len(data))
