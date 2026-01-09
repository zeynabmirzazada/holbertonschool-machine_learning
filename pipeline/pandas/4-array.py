#!/usr/bin/env python3
'''
function that takes dataframe and converts last 10 rows into numpy ndarray
'''


def array(df):
    '''
    select only High and Close columns
    convert last 10 rows into ndarray
    '''
    df = df[['High', 'Close']]
    df = df.tail(10)
    df = df.to_numpy()
    return df
