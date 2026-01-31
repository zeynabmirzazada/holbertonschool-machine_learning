#!/usr/bin/env python3
"""
Module to calculate the definiteness of a matrix
"""
import numpy as np


def definiteness(matrix):
    """
    Calculates the definiteness of a matrix.

    Args:
        matrix: a numpy.ndarray of shape (n, n)

    Returns:
        String indicating definiteness or None
    """
    # 1. Validation: Must be a numpy ndarray
    if not isinstance(matrix, np.ndarray):
        raise TypeError("matrix must be a numpy.ndarray")

    # 2. Validation: Must be square and non-empty
    if len(matrix.shape) != 2 or matrix.shape[0] != matrix.shape[1]:
        return None
    if matrix.size == 0:
        return None

    # 3. Check for symmetry (Definiteness is defined for symmetric matrices)
    if not np.allclose(matrix, matrix.T):
        return None

    try:
        # Calculate eigenvalues
        eigenvalues = np.linalg.eigvals(matrix)

        pos = np.any(eigenvalues > 1e-10)
        neg = np.any(eigenvalues < -1e-10)
        zero = np.any(np.isclose(eigenvalues, 0, atol=1e-10))

        if pos and not neg and not zero:
            return "Positive definite"
        if pos and not neg and zero:
            return "Positive semi-definite"
        if neg and not pos and not zero:
            return "Negative definite"
        if neg and not pos and zero:
            return "Negative semi-definite"
        if pos and neg:
            return "Indefinite"

    except np.linalg.LinAlgError:
        return None

    return None
