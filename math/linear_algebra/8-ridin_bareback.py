#!/usr/bin/env python3
"""
Module that provides a function to perform matrix multiplication.
"""


def mat_mul(mat1, mat2):
    """
    Performs matrix multiplication between two 2D matrices.

    Args:
        mat1 (list of lists): First 2D matrix
        mat2 (list of lists): Second 2D matrix

    Returns:
        list of lists: New matrix resulting from multiplication,
        or None if multiplication is not possible
    """
    if not isinstance(mat1, list) or not isinstance(mat2, list):
        return None

    if not mat1 or not mat2:
        return None

    if len(mat1[0]) != len(mat2):
        return None

    result = []
    for i in range(len(mat1)):
        row = []
        for j in range(len(mat2[0])):
            total = 0
            for k in range(len(mat2)):
                total += mat1[i][k] * mat2[k][j]
            row.append(total)
        result.append(row)

    return result
