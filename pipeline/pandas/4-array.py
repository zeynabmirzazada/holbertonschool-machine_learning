#!/usr/bin/env python3
'''
function that takes dataframe and converts last 10 rows into numpy ndarray
'''
import pandas as pd


def array(df):
    '''
    select only High and Close columns
    convert last 10 rows into ndarray
    '''
    return df[['High', 'Close']].tail(10).to_numpy()
