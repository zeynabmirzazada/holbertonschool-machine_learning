#!/usr/bin/env python3
"""
Module that provides a function to perform matrix multiplication.
"""

import numpy as np


def np_matmul(mat1, mat2):
    """
    Performs matrix multiplication between two numpy arrays.

    Args:
        mat1 (numpy.ndarray): First matrix
        mat2 (numpy.ndarray): Second matrix

    Returns:
        numpy.ndarray: Resulting matrix product
    """
    return np.matmul(mat1, mat2)
