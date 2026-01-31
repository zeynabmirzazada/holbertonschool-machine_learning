#!/usr/bin/env python3
"""
Module to calculate the minor matrix of a square matrix
"""


def determinant(matrix):
    """
    Helper function to calculate determinant of a matrix recursively.
    """
    n = len(matrix)
    if n == 0:
        return 1
    if n == 1:
        return matrix[0][0]
    if n == 2:
        return (matrix[0][0] * matrix[1][1]) - (matrix[0][1] * matrix[1][0])

    det = 0
    for j in range(n):
        sub_matrix = [row[:j] + row[j+1:] for row in matrix[1:]]
        det += ((-1) ** j) * matrix[0][j] * determinant(sub_matrix)
    return det


def inverse(matrix):
    """
    Calculates the minor matrix of a matrix.

    Args:
        matrix: a list of lists whose minor matrix should be calculated

    Returns:
        The minor matrix of the input matrix
    """
    # 1. Validation: Must be a list of lists
    if not isinstance(matrix, list) or len(matrix) == 0:
        raise TypeError("matrix must be a list of lists")
    if not all(isinstance(row, list) for row in matrix):
        raise TypeError("matrix must be a list of lists")

    # 2. Validation: Must be non-empty and square
    n = len(matrix)
    if n == 1 and len(matrix[0]) == 0:
        raise ValueError("matrix must be a non-empty square matrix")
    if not all(len(row) == n for row in matrix):
        raise ValueError("matrix must be a non-empty square matrix")

    # 4. Calculate Minor Matrix
    minor_matrix = []
    for i in range(n):
        row_minors = []
        for j in range(n):
            # Create sub-matrix by removing row i and column j
            sub_matrix = [row[:j] + row[j+1:] for k, row in enumerate(matrix)
                          if k != i]
            row_minors.append(((-1) ** (i + j)) * determinant(sub_matrix))
        minor_matrix.append(row_minors)

    det_ = determinant(matrix)
    if det_ == 0:
        return None

    # 5. Handle 1x1 case
    if n == 1:
        return [[1 / matrix[0][0]]]

    adjugate_matrix = []
    for a in range(len(minor_matrix)):
        row_ = []
        for b in range(len(minor_matrix[0])):
            row_.append(minor_matrix[b][a])
        adjugate_matrix.append(row_)

    inv_matrix = [[col / det_ for col in row] for row in adjugate_matrix]
    return inv_matrix
