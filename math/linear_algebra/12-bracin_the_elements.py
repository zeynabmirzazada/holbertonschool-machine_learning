#!/usr/bin/env python3
"""
Module that provides a function to perform element-wise operations.
"""


def np_elementwise(mat1, mat2):
    """
    Performs element-wise addition, subtraction, multiplication,
    and division on two numpy arrays.

    Args:
        mat1 (numpy.ndarray): First input
        mat2 (numpy.ndarray or scalar): Second input

    Returns:
        tuple: Element-wise sum, difference, product, and quotient
    """
    return (mat1 + mat2, mat1 - mat2, mat1 * mat2, mat1 / mat2)
