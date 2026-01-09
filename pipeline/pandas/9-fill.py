#!/usr/bin/env python3
'''
function that takes a pd.DataFrame and removes the Weighted_Price column.
Fills missing values in the Close column with the previous rows value. Fills
missing values in the High, Low, and Open columns with the corresponding Close
value in the same row. Sets missing values in Volume_(BTC) and
Volume_(Currency) to 0. Returns: the modified pd.DataFrame.
'''


def fill(df):
    '''
    function that takes a pd.DataFrame and removes the Weighted_Price column.
    Fills missing values in the Close column with the previous rows value.
    Fills missing values in the High, Low, and Open columns with the
    corresponding Close value in the same row. Sets missing values in
    Volume_(BTC) and Volume_(Currency) to 0. Returns: the modified
    pd.DataFrame.
    '''
    df.drop(columns=['Weighted_Price'], inplace=True)
    df['Close'] = df['Close'].ffill()
    df[['High', 'Low', 'Open']] = df[['High', 'Low', 'Open']].fillna(df['Close'])
    df['Volume_(BTC)'] = df['Volume_(BTC)'].fillna(0)
    df['Volume_(Currency)'] = df['Volume_(Currency)'].fillna(0)
    return df
