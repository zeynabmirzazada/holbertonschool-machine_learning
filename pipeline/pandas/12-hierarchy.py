#!/usr/bin/env python3
'''
function that takes two pd.DataFrame objects and rearranges the MultiIndex so
that Timestamp is the first level. Concatenates the bitstamp and coinbase
tables from timestamps 1417411980 to 1417417980, inclusive. Adds keys to the
data, labeling rows from df2 as bitstamp and rows from df1 as coinbase.
Ensures the data is displayed in chronological order.
You should use index = __import__('10-index').index.
Returns: the concatenated pd.DataFrame.
'''
import pandas as pd
index = __import__('10-index').index


def hierarchy(df1, df2):
    '''
    function that takes two pd.DataFrame objects and rearranges the MultiIndex
    that Timestamp is the first level. Concatenates the bitstamp and coinbase
    tables from timestamps 1417411980 to 1417417980, inclusive. Adds keys to
    data, labeling rows from df2 as bitstamp and rows from df1 as coinbase.
    '''
    df1 = index(df1)
    df2 = index(df2)
    return pd.concat([df2.loc[1417411980:1417417980],
        df1.loc[1417411980:1417417980]],
        keys=['bitstamp', 'coinbase']).sort_index((
