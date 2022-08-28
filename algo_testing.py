import pandas as pd
import numpy as np
import plotly.graph_objects as go
import pandas_ta as ta

df = pd.read_csv('ETF_data.csv')

assets = ['TQQQ', 'SQQQ', 'SPY', 'UVXY', 'SPXL', 'BSV', 'TECL']

df['TQQQ_RSI'] = ta.rsi(df['TQQQ'], length=10)
df['SPY_RSI'] = ta.rsi(df['SPY'], length=10)
df['UVXY_RSI'] = ta.rsi(df['UVXY'], length=10)
df['SQQQ_RSI'] = ta.rsi(df['SQQQ'], length=10)
df['BSV_RSI'] = ta.rsi(df['BSV'], length=10)
df['SPXL_RSI'] = ta.rsi(df['SPXL'], length=10)
df['SPY_SMA'] = ta.sma(df['SPY'], length=200)
df['TQQQ_SMA'] = ta.sma(df['TQQQ'], length=20)

df = df[-2400:]

# print(df)

#get all the candles and calculate the price changes
price_changes = []
prices = []
times = []
ta_values = []

# print(df)
for index, row in df.iterrows():
    prices.append(np.array([1] + [row[asset] for asset in assets]))
    ta_values.append(np.array([row['TQQQ_RSI'], row['SPY_RSI'], row['UVXY_RSI'], row['SQQQ_RSI'], row['BSV_RSI'], row['SPXL_RSI'], row['SPY_SMA'], row['TQQQ_SMA']]))
    times.append(row['Date'])

    if len(prices) == 1:
        price_changes.append(np.ones(len(assets) + 1))
    else:
        price_changes.append(prices[-1] / prices[-2])


# Set variables:
predictions_list = []
returns = []
previous_prices = []
previous_price_changes = []

target = 0
previous_target = 0
previous_change = price_changes[0]

for index, (change, price, ta_value) in enumerate(zip(price_changes, prices, ta_values)):

    previous_prices.append(price)
    previous_price_changes.append(change)

    # print(change)
    # print(price)
    # continue

    # CALCULATE RETURNS:

    fee = 0.2 # Fees/slippage percentage % Buy and Sell

    activate_stop_loss = False

    # Fees:
    if target == previous_target:

        if activate_stop_loss == True:
            stop_loss = 0.9
            if change[target] <= stop_loss:
                change[target] = stop_loss

        returns.append(change[target])

    elif target == 0:
        returns.append(1)
    
    else:

        if activate_stop_loss == True:
            stop_loss = 0.9
            if change[target] <= stop_loss:
                change[target] = stop_loss

        returns.append((change[target] * (1-(fee / 100))))

    # ALGO LOGIC:

    # --------------------------------------------------------------------------------------#
    # IF current price of SPY is greater than 200d MA of price of SPY:
    # |-> IF 10d RSI of TQQQ is greater than 80:
    # |   |-> BUY UVXY
    # |   ELSE:
    # |   |-> IF 10d RSI of SPXL is greater than 80:
    # |   |   |-> BUY UVXY
    # |   |   ELSE:
    # |   |   |-> BUY TQQQ
    # ELSE:
    # |-> IF 10d RSI of TQQQ is less than 30:
    # |   |-> BUY TECL
    # |   ELSE:
    # |   |-> IF 10d RSI of SPY is less than 30:
    # |   |   |-> BUY SPXL
    # |   |   ELSE:
    # |   |   |-> IF 10d RSI of UVXY is greater than 75:
    # |   |   |   |-> IF 10d RSI of UVXY is greater than 85:
    # |   |   |   |   |-> IF current price of TQQQ is greater than 20d MA of price of TQQQ:
    # |   |   |   |   |   |-> BUY TQQQ
    # |   |   |   |   |   ELSE:
    # |   |   |   |   |   |-> SORT 10d RSI of SQQQ and BSV - SELECT Top 1:
    # |   |   |   |   |   |   |-> BUY SQQQ or BSV
    # |   |   |   |   ELSE:
    # |   |   |   |   |-> BUY UVXY
    # |   |   |   ELSE:
    # |   |   |   |-> IF current price of TQQQ is greater than 20d MA of price of TQQQ:
    # |   |   |   |   |-> BUY TQQQ
    # |   |   |   |   ELSE:
    # |   |   |   |   |-> SORT 10d RSI of SQQQ and BSV - SELECT Top 1:
    # |   |   |   |   |   |-> BUY SQQQ or BSV
    # --------------------------------------------------------------------------------------#

    if price[3] > ta_value[6]:
        if ta_value[0] > 80:
            prediction = 4 # UVXY
        else:
            if ta_value[5] > 80:
                prediction = 4 # UVXY
            else:
                prediction = 1 # TQQQ
    else:
        if ta_value[0] < 30:
            prediction = 7 # TECL
        else:
            if ta_value[1] < 30:
                prediction = 5 # SPXL
            else:
                if ta_value[2] > 75:
                    if ta_value[2] > 85:
                        if price[1] > ta_value[7]:
                            prediction = 1 # TQQQ
                        else:
                            sort = {2: ta_value[3], 0: ta_value[4]}
                            prediction = max(sort, key=sort.get) # SQQQ or BSV
                    else:
                        prediction = 4 # UVXY
                else:
                    if price[1] > ta_value[7]:
                        prediction = 1
                    else:
                        sort = {2: ta_value[3], 0: ta_value[4]}
                        prediction = max(sort, key=sort.get) # SQQQ or BSV

    # STORE INFO:

    # Store previous index prediction
    predictions_list.append(prediction)

    # Store previous targets and elements
    previous_target = target
    previous_change = change

    # Target is the predicted coin index
    target = prediction


# ALGO STATS:

print('Return: %.2f' % np.prod(returns))
print('Avg mean return: %.2f %%' % ((np.mean(returns)-1)*100))
print('Winrate: %.2f %%' % ((len([x for x in returns if x >= 1])/len(returns))*100))


# Make DataFrame of Dates, Total Return
prices_df = pd.DataFrame({'Date':times, 'Profit':returns})

# Set Date as index and make datetimeobject
prices_df.set_index('Date', inplace=True)
prices_df.index = pd.to_datetime(prices_df.index)


Roll_Max = prices_df['Profit'].cummax()
Daily_Drawdown = prices_df['Profit']/Roll_Max - 1.0
Max_Daily_Drawdown = Daily_Drawdown.cummin()
# average_drawdown = Max_Daily_Drawdown.mean()

print('Max Drawdown: %.2f %%' % ((Max_Daily_Drawdown.min())*100))

prices_df['Return'] = prices_df['Profit'] - 1

sharpe_ratio = prices_df['Return'].mean() / prices_df['Return'].std() * np.sqrt(730)
print('Sharpe Ratio: %.2f' % (sharpe_ratio))

yearly_avg = prices_df.groupby(pd.Grouper(freq='Y'))['Profit'].prod()
quarterly_avg = prices_df.groupby(pd.Grouper(freq='Q'))['Profit'].prod()
monthly_avg = prices_df.groupby(pd.Grouper(freq='M'))['Profit'].prod()
weekly_avg = prices_df.groupby(pd.Grouper(freq='W'))['Profit'].prod()
daily_avg = prices_df.groupby(pd.Grouper(freq='D'))['Profit'].prod()

print('Yearly Return: %.2f %%' % ((yearly_avg.mean()-1)*100))
# print('Quarterly Return: %.2f %%' % ((quarterly_avg.mean()-1)*100))
print('Monthly Return: %.2f %%' % ((monthly_avg.mean()-1)*100))
# print('Weekly Return: %.2f %%' % ((weekly_avg.mean()-1)*100))
print('Daily Return: %.2f %%' % ((daily_avg.mean()-1)*100))


value = 1
values = []

for i in returns:
    value += (i-1) * value
    values.append(value)


bah_value = 1
bah_values = []
bah_returns = []

for i in price_changes:
    bah_value += (i[1]-1) * bah_value
    bah_values.append(bah_value)
    bah_returns.append(i[1])

prices_df['BAH'] = bah_returns

yearly_avg_bah = prices_df.groupby(pd.Grouper(freq='Y'))['BAH'].prod()
quarterly_avg_bah = prices_df.groupby(pd.Grouper(freq='Q'))['BAH'].prod()
monthly_avg_bah = prices_df.groupby(pd.Grouper(freq='M'))['BAH'].prod()

print('BAH Yearly Return: %.2f %%' % ((yearly_avg_bah.mean()-1)*100))
# print('BAH Quarterly Return: %.2f %%' % ((quarterly_avg_bah.mean()-1)*100))
print('BAH Monthly Return: %.2f %%' % ((monthly_avg_bah.mean()-1)*100))


# PLOTTING:

fig = go.Figure()

fig.add_trace(go.Scatter(x=times, y=values, name='ALGO'))
fig.add_trace(go.Scatter(x=times, y=bah_values, name='TQQQ'))

# for i, asset in enumerate(['USD'] + assets):
#     # fig.add_trace(go.Scatter(x=times, y=[p[i] for p in portfolios[1:]], stackgroup='one', name=asset, line=dict(width=0, color=cols[i]), showlegend=False), row=2, col=1)
#     if asset == 'USD':
#         continue
#     fig.add_trace(go.Scatter(x=times, y=[p[i]/prices[0][i] for p in prices[1:]], name=asset))

fig.update_xaxes(rangebreaks=[dict(bounds=["sat", "mon"])])
# fig.layout.yaxis.tickformat = ',.0%'
fig.update_yaxes(type='log')

fig.show()


bar_fig = go.Figure(go.Bar(x=quarterly_avg.index, y=((quarterly_avg-1)*100), name='ALGO'))
bar_fig.add_trace(go.Bar(x=quarterly_avg_bah.index, y=((quarterly_avg_bah-1)*100), name='TQQQ'))

bar_fig.add_hline(y=((quarterly_avg.mean()-1)*100), line_dash="dash", line_color="blue", annotation_text="{:.2f} %".format((quarterly_avg.mean()-1)*100), annotation_position="top left")
bar_fig.add_hline(y=((quarterly_avg_bah.mean()-1)*100), line_dash="dash", line_color="red", annotation_text="{:.2f} %".format((quarterly_avg_bah.mean()-1)*100), annotation_position="top left")

# bar_fig.show()
