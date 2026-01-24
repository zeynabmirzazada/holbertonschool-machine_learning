#!/usr/bin/env python3
"""
Module that provides a function to concatenate two numpy arrays.
"""

import numpy as np


def np_cat(mat1, mat2, axis=0):
    """
    Concatenates two numpy arrays along a specific axis.

    Args:
        mat1 (numpy.ndarray): First matrix
        mat2 (numpy.ndarray): Second matrix
        axis (int): Axis along which to concatenate

    Returns:
        numpy.ndarray: New concatenated array
    """
    return np.concatenate((mat1, mat2), axis=axis)
