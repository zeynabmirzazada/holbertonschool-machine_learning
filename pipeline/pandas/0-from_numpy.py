#!/usr/bin/env python3
"""
0-from_numpy module

Creates a pandas DataFrame from a NumPy ndarray.
"""

import pandas as pd


def from_numpy(array):
    """
    Creates a pandas DataFrame from a NumPy ndarray.

    The columns are labeled in alphabetical order using
    capital letters (A–Z).

    Args:
        array (numpy.ndarray): Input NumPy array

    Returns:
        pandas.DataFrame: DataFrame with labeled columns
    """
    n_cols = array.shape[1]
    columns = [chr(65 + i) for i in range(n_cols)]
    return pd.DataFrame(array, columns=columns)
