#!/usr/bin/env python3
'''
Computes descriptive statistics for all columns except the Timestamp column
'''


def analyze(df):
    '''
    Computes descriptive statistics for all columns except the Timestamp colum
    '''
    return df.drop(columns=['Timestamp']).describe()
