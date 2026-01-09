#!/usr/bin/env python3
'''
function that loads data from a file as dataframe
'''
import pandas as pd
def from_file(filename, delimiter):
    '''
    Args:
        filename - is the file to load from
        delimiter - is the column separator
    '''
    df = pd.read_csv(filename, delimiter=delimiter)
    return df
