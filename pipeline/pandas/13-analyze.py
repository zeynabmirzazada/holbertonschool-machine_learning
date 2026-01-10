#!/usr/bin/env python3
'''
Computes descriptive statistics for all columns except the Timestamp column
'''
import pandas as pd


def analyze(df):
    '''
    Computes descriptive statistics for all columns except the Timestamp colum
    '''
    return df.describe(exclude=[int64])
