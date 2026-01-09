#!/usr/bin/env python3
'''
function that takes dataframe, sorts it by High price in descending order, and
returns sorted df
'''


def high(df):
    '''
    function that takes dataframe, sorts it by High price in descending order,
    and returns sorted df
    '''
    return df.sort_values(by='High', ascending=False)
