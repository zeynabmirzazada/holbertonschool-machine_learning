#!/usr/bin/env python3
'''
function that takes dataframe and sorts the data in reverse chronological
order, and then returns transposed dataframe
'''


def flip_switch(df):
    '''
    function that takes dataframe and sorts the data in reverse chronological
    order, and then returns transposed dataframe
    '''
    return df.sort_values(by='Timestamp', ascending=False).T
