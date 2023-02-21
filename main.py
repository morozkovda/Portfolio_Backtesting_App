import yfinance as yf
from pandas_datareader import data as pdr
from datetime import datetime
import pandas as pd
import numpy as np
#pip install git+https://github.com/mementum/backtrader.git@0fa63ef4a35dc53cc7320813f8b15480c8f85517#egg=backtrader
import backtrader as bt
import backtrader.analyzers as btanalyzers
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from backtrader import Analyzer, TimeFrame
import time
yf.pdr_override()

# get data
tickers = ['BTC-USD' ,'ETH-USD', 'BNB-USD']
stock = pdr.get_data_yahoo(tickers,
                start= datetime(2020, 1, 1),
                end= datetime(2023, 2, 16), interval = '1d')

# create function to process data for cerebro
very_small_float = 1.0
idx = pd.IndexSlice
def getStock(df, s):
    c = df.loc[:,idx[:,s]]
    c.columns = c.columns.droplevel(1)
    c = pd.DataFrame(c.to_records()).set_index('Date')
    c = c.fillna(very_small_float)
    return c
def getReturnAsDataFrame(pfa):
    t = pfa.get_analysis()
    listst = sorted(t.items()) # sorted by key, return a list of tuples
    x, y = zip(*listst) # unpack a list of pairs into two tuples
    dd = {'data':x, 'Portfolio':y}
    df = pd.DataFrame(dd).set_index('data')
    return df


# import strategy
from backtest import optimizer, Model

# launch cerebro
cerebro = bt.Cerebro()

# set initial cash
cerebro.broker.set_cash(100000)
print('Starting Portfolio Value: %.2f' % cerebro.broker.getvalue())

# add data to cerebro datafeed
for a in tickers:
    cerebro.adddata(bt.feeds.PandasData(dataname=getStock(stock, a), name=a))

#add analyzers
cerebro.addanalyzer(bt.analyzers.TimeReturn, timeframe=bt.TimeFrame.Months, _name='PortfolioReturns')
cerebro.addanalyzer(btanalyzers.SharpeRatio, riskfreerate=0.0, timeframe=bt.TimeFrame.Months, _name='mysharpe')
cerebro.addanalyzer(btanalyzers.Returns, _name='myreturn')
cerebro.addanalyzer(btanalyzers.DrawDown, _name='mydrawdown')
cerebro.addanalyzer(bt.analyzers.AnnualReturn, _name='_AnnualReturn')


# set params
params = {'DataCounter' : 125,
        'RebalanceDay': 22,
        'printlog': 0,
        'model': Model(),
         'model_params':
              {'model_n': 'equal',
               'isCleanWeight': False}
          }

# add strategy
cerebro.addstrategy(optimizer, **params)

# run strategy
thestrats = cerebro.run()

thestrat = thestrats[0]

#analysis module
ret =  {'Max_Drawdown':thestrat.analyzers.getbyname('mydrawdown').get_analysis()['max']['drawdown'],
        'CAGR':thestrat.analyzers.getbyname('myreturn').get_analysis()['rnorm100'],
        'Annual Return' : thestrat.analyzers._AnnualReturn.get_analysis(),
        'Sharpe_Ratio':thestrat.analyzers.getbyname('mysharpe').get_analysis()['sharperatio'],
        'Value': cerebro.broker.getvalue()
        }

for key, value in ret.items():
    print("--------------- %s -----------------" %key)
    print(value)

df = getReturnAsDataFrame(thestrat.analyzers.getbyname('PortfolioReturns'))
df.to_csv('data/model_return.csv')

# plot and print results (note: to plot the results you need to install specific version of bt pip specified above)
cerebro.plot(figsize=(230,130))



print('Final Portfolio Value: %.2f' % cerebro.broker.getvalue())