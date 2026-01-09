#!/usr/bin/env python3
'''
function that takes df and removes any entries where Close has NaN values
'''


def prune(df):
    '''
    function that takes df and removes any entries where Close has NaN values
    '''
    return df.dropna(subset=['Close'])
