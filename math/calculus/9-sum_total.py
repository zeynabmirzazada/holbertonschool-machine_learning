#!/usr/bin/env python3
"""
Module for calculating the sum of i squared from 1 to n
"""


def summation_i_squared(n):
    """
    Calculates the sum of i^2 for i in range 1 to n using
    the square pyramidal number formula.

    Args:
        n (int): The stopping condition.

    Returns:
        int: The sum of i^2 from 1 to n.
        None: If n is not a valid number.
    """
    if not isinstance(n, (int, float)) or n < 1:
        return None

    # Using the formula n(n+1)(2n+1) / 6
    # We use integer division // to ensure the return is an integer
    result = (n * (n + 1) * (2 * n + 1)) // 6

    return int(result)
