#!/usr/bin/env python3
'''
function that takes a pd.DataFrame and extracts the columns High, Low, Close,
and Volume_BTC. Then selects every 60th row from these columns.
Returns: the sliced pd.DataFrame
'''


def slice(df):
    '''
    df - dataframe
    return: sliced df with every 60th row
    '''
    df = df[['High', 'Low', 'Close', 'Volume_BTC']]
    return df[59::60]
