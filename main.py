import yfinance as yf
from pandas_datareader import data as pdr
from datetime import datetime
import numpy as np
import pandas as pd
#pip install git+https://github.com/mementum/backtrader.git@0fa63ef4a35dc53cc7320813f8b15480c8f85517#egg=backtrader
import backtrader as bt
from pypfopt import HRPOpt, risk_models, expected_returns
yf.pdr_override()

tickers = ['BTC-USD' ,'ETH-USD', 'BNB-USD']
stock = pdr.get_data_yahoo(tickers,
                start= datetime(2020, 1, 1),
                end= datetime(2023, 2, 16), interval = '1d')
stock.head()

very_small_float = 1.0
idx = pd.IndexSlice
def getStock(df, s):
    c = df.loc[:,idx[:,s]]
    c.columns = c.columns.droplevel(1)
    c = pd.DataFrame(c.to_records()).set_index('Date')
    c = c.fillna(very_small_float)
    return c

from backtest import optimizer

cerebro = bt.Cerebro()

cerebro.broker.set_cash(100000)
print('Starting Portfolio Value: %.2f' % cerebro.broker.getvalue())


for a in tickers:
    cerebro.adddata(bt.feeds.PandasData(dataname=getStock(stock, a), name=a))

cerebro.addstrategy(optimizer)

thestrats = cerebro.run()
thestrat = thestrats[0]

cerebro.plot()

print('Final Portfolio Value: %.2f' % cerebro.broker.getvalue())