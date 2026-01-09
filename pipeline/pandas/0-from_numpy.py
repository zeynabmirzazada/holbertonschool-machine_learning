#!/usr/bin/env python3
import pandas as pd
"""
importing pandas library to be able to use dataframe
"""


def from_numpy(array):

    """
    Creates a pandas DataFrame from a NumPy ndarray.

    Columns are labeled with capital letters in alphabetical order.
    """
    n_cols = array.shape[1]
    columns = [chr(65 + i) for i in range(n_cols)]
    return pd.DataFrame(array, columns=columns)
