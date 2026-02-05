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

    def pdf(self, x):
        """
        Calculates the value of the PDF for a given time period.

        Args:
            x (float): The time period.

        Returns:
            float: The PDF value for x.
        """
        if x < 0:
            return 0

        # Constants
        e = 2.7182818285
        lambtha = self.lambtha

        # PDF formula: lambtha * e^(-lambtha * x)
        return lambtha * (e ** (-lambtha * x))

    def cdf(self, x):
        """
        calculates the value of the CDF for a given time period

        Args:
            x(float): the time period

        Returns:
            float: CDF
        """
        if x < 0:
            return 0

        # Constants
        e = 2.7182818285
        lambtha = self.lambtha

        # CDF formula: 1 - e^(-lambtha * x)
        return 1 - e ** (-lambtha * x)
