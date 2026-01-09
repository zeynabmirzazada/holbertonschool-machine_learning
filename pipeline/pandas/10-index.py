#!/usr/bin/env python3
'''
Sets the Timestamp column as the index of the dataframe.
'''


def index(df):
    """
    Sets the Timestamp column as the index of the dataframe.
    """
    # Use set_index to move 'Timestamp' to the index
    # Note: set_index returns a new DataFrame by default
    df = df.set_index('Timestamp')
    return df
