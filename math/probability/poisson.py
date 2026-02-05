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

    def pmf(self, k):
        """
        Calculates the value of the PMF for a given number of successes.

        Args:
            k (int): The number of successes.

        Returns:
            float: The PMF value for k.
        """
        # Convert k to integer as per requirements
        k = int(k)

        # Poisson distribution is defined for k >= 0
        if k < 0:
            return 0

        # Constants
        e = 2.7182818285
        lambtha = self.lambtha

        # Calculate k! (factorial)
        factorial = 1
        for i in range(1, k + 1):
            factorial *= i

        # Calculate PMF: (e^-lambtha * lambtha^k) / k!
        # Note: x**y is the power operator in Python
        pmf_value = (e ** (-lambtha) * (lambtha ** k)) / factorial

        return pmf_value
    def cdf(self, k):
        """
        Calculates the value of the CDF for a given number of successes.

        Args:
            k (int): The number of successes.

        Returns:
            float: The CDF value for k.
        """
        k = int(k)
        if k < 0:
            return 0

        # Summation of PMF from 0 to k
        cdf_value = 0
        for i in range(k + 1):
            cdf_value += self.pmf(i)

        return cdf_value
