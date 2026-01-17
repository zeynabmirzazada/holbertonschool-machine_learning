#!/usr/bin/env python3
"""
Module for calculating the integral of a polynomial.
"""


def poly_integral(poly, C=0):
    """
    Calculates the integral of a polynomial.

    Args:
        poly (list): A list of coefficients representing a polynomial.
                     The index represents the power of x.
        C (int): The integration constant. Defaults to 0.

    Returns:
        list: Coefficients of the integral polynomial.
        None: If poly or C are not valid.
    """
    # Validate that poly is a list of numbers (int or float)
    if not isinstance(poly, list) or len(poly) == 0:
        return None
    if not all(isinstance(coeff, (int, float)) for coeff in poly):
        return None

    # Validate that the integration constant C is an integer
    if not isinstance(C, int):
        return None

    # Handle the case of an empty-equivalent polynomial [0]
    if poly == [0]:
        return [C]

    # Calculate the integral: integral[n+1] = poly[n] / (n + 1)
    # The integration constant C becomes the coefficient at index 0
    integral = [C]
    for i in range(len(poly)):
        val = poly[i] / (i + 1)
        # Convert to integer if it is a whole number (e.g., 5.0 -> 5)
        if val % 1 == 0:
            val = int(val)
        integral.append(val)

    # Ensure the returned list is as small as possible by removing trailing 0s
    # but maintaining at least the constant term
    while len(integral) > 1 and integral[-1] == 0:
        integral.pop()

    return integral
