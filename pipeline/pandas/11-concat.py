#!/usr/bin/env python3
'''
function that takes 2 df objects, indexes them on Timestamp column,
concatenates df2 to the top of df1, adds keys to the concatenated data,
labeling the rows from df2 as bitstamp and the rows from df1 as coinbase,
returns the concatenated df
'''
index = __import__('10-index').index


def concat(df1, df2):
    '''
    function that takes 2 df objects, indexes them on Timestamp column,
    concatenates df2 to the top of df1, adds keys to the concatenated data,
    labeling the rows from df2 as bitstamp and the rows from df1 as coinbase,
    returns the concatenated df
    '''
    df1 = index(df1)
    df2 = index(df2)
    df2 = df2.loc[:1417411920]
    return pd.concat([df2, df1], keys=['bitstamp', 'coinbase'])
