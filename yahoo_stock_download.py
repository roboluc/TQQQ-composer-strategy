import pandas as pd
from pandas_datareader import data
from datetime import datetime


# Define the instruments to download.
tickers = ['TQQQ', 'SQQQ', 'SPY', 'UVXY', 'SPXL', 'BSV', 'TECL']

# Define the daterange
start_date = '2012-01-01'
end_date = '2023-01-01'

# User pandas_reader.data.DataReader to load the desired data. As simple as that.
panel_data = data.DataReader(tickers, 'yahoo', start_date, end_date)

# panel_data = panel_data[['Open', 'High', 'Low', 'Close', 'Volume', 'Adj Close']]

# print(panel_data[['Open', 'High', 'Low', 'Close', 'Volume']])

# print(panel_data)

print(panel_data['Adj Close'])

panel_data['Adj Close'].to_csv('ETF_data.csv', sep=',')
