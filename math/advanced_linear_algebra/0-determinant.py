#!/usr/bin/env python3
"""
Module to calculate the determinant of a matrix
"""


def determinant(matrix):
    """
    Calculates the determinant of a matrix recursively.

    Args:
        matrix: a list of lists whose determinant should be calculated

    Returns:
        The determinant of the matrix
    """
    # 1. Validation: Must be a list of lists
    if not isinstance(matrix, list) or len(matrix) == 0:
        if matrix == []:
            return 1
        raise TypeError("matrix must be a list of lists")

    if not all(isinstance(row, list) for row in matrix):
        raise TypeError("matrix must be a list of lists")

    # 2. Handle 0x0 matrix
    if matrix == [[]]:
        return 1

    # 3. Validation: Must be square
    n = len(matrix)
    if not all(len(row) == n for row in matrix):
        raise ValueError("matrix must be a square matrix")

    # 4. Base Case: 1x1 matrix
    if n == 1:
        return matrix[0][0]

    # 5. Base Case: 2x2 matrix (optimization)
    if n == 2:
        return (matrix[0][0] * matrix[1][1]) - (matrix[0][1] * matrix[1][0])

    # 6. Recursive Step: Laplace Expansion along the first row
    det = 0
    for j in range(n):
        # Create the sub-matrix (minor) by removing row 0 and column j
        sub_matrix = [row[:j] + row[j+1:] for row in matrix[1:]]
        # Recursively add/subtract the cofactor
        det += ((-1) ** j) * matrix[0][j] * determinant(sub_matrix)

    return det
