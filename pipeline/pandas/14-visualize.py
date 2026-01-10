#!/usr/bin/env python3

import matplotlib.pyplot as plt
import pandas as pd
from_file = __import__('2-from_file').from_file

df = from_file('coinbaseUSD_1-min_data_2014-12-01_to_2019-01-09.csv', ',')
df.drop(columns=['Weighted_Price'], inplace=True)
df = df.rename(columns={'Timestamp': 'Date'})
df['Date'] = pd.to_datetime(df['Date'], unit='d')
df = df.set_index('Date')
df['Close'] = df['Close'].ffill()
for col in ['High', 'Low', 'Open']:
    df[col] = df[col].fillna(df['Close'])
for col in ['Volume_(BTC)', 'Volume_(Currency)']:
    df[col] = df[col].fillna(0)
df = df.loc['2017-01-01':]
df = df.resample('D').agg({
    'High': 'max',
    'Low': 'min',
    'Open': 'mean',
    'Close': 'mean',
    'Volume_(BTC)': 'sum',
    'Volume_(Currency)': 'sum'
})

# Display the transformed dataframe as requested
print(df)

# 7. Plot the data
df.plot()
plt.show()
