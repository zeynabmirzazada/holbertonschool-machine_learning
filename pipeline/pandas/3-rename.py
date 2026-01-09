#!/usr/bin/env python3
'''
function that takes DataFrame as an input, and renames one column, converts
timestamp to datetime and shows only 2 columns
'''
import pandas as pd


def rename(df):
    '''
    df is DataFrame
    return will be modified df
    '''
    df.rename(columns={'Timestamp': 'Datetime'}, inplace=True)
    df['Datetime'] = pd.to_datetime(df['Datetime'], unit='s')
    return df[['Datetime'], ['Close']]
