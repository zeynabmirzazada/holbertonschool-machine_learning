#!/usr/bin/env python3
"""
Module for calculating the derivative of a polynomial.
"""


def poly_derivative(poly):
    """
    Calculates the derivative of a polynomial.

    Args:
        poly (list): A list of coefficients representing a polynomial.
                     The index represents the power of x.

    Returns:
        list: Coefficients of the derivative polynomial.
        None: If poly is not a valid list of coefficients.
    """
    # Validate that poly is a non-empty list of numbers
    if not isinstance(poly, list) or len(poly) == 0:
        return None
    for coeff in poly:
        if not isinstance(coeff, (int, float)):
            return None

    # A polynomial of a single constant (e.g., [5]) has a derivative of 0
    if len(poly) == 1:
        return [0]

    # Calculate derivative: derivative[n-1] = poly[n] * n
    # We skip the constant term (index 0) because its derivative is 0
    derivative = [poly[i] * i for i in range(1, len(poly))]

    # Special case: if the polynomial was [0, 0, 0], derivative is [0]
    # This ensures we don't return an empty list if all terms were zero
    if not any(derivative):
        return [0]

    return derivative
