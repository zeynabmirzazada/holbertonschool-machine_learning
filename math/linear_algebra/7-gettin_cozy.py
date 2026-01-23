#!/usr/bin/env python3
"""
Module that provides a function to concatenate two 2D matrices.
"""


def cat_matrices2D(mat1, mat2, axis=0):
    """
    Concatenates two 2D matrices along a specific axis.

    Args:
        mat1 (list of lists): First 2D matrix
        mat2 (list of lists): Second 2D matrix
        axis (int): Axis along which to concatenate (0 or 1)

    Returns:
        list of lists: New concatenated matrix, or None if impossible
    """
    if not isinstance(mat1, list) or not isinstance(mat2, list):
        return None

    if axis == 0:
        if not mat1:
            return [row[:] for row in mat2]
        if not mat2:
            return [row[:] for row in mat1]

        if len(mat1[0]) != len(mat2[0]):
            return None

        return [row[:] for row in mat1] + [row[:] for row in mat2]

    if axis == 1:
        if len(mat1) != len(mat2):
            return None

        new_matrix = []
        for row1, row2 in zip(mat1, mat2):
            new_matrix.append(row1[:] + row2[:])

        return new_matrix

    return None
