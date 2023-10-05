import yfinance as yf
from pandas_datareader import data as pdr
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
#pip install git+https://github.com/mementum/backtrader.git@0fa63ef4a35dc53cc7320813f8b15480c8f85517#egg=backtrader
import backtrader as bt
import backtrader.analyzers as btanalyzers
from tqdm import tqdm
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from backtrader import Analyzer, TimeFrame
import time
from graphs import plotReturns,plotStatsValues
import streamlit as st
yf.pdr_override()

# create function to process returns_data for cerebro
idx = pd.IndexSlice
def getStock(df, s):
    c = df.loc[:,idx[:,s]]
    c.columns = c.columns.droplevel(1)
    c = pd.DataFrame(c.to_records()).set_index('Date')
    if c.isna().any().any():
        raise ValueError("Array contains NaN values which means that there is no available data for some asset during some part of chosen period"
                         " also you cannot mix shares with crypto assets as shares are not traded on weekends")
    return c
def getReturnAsDataFrame(params,pfa):
    t = pfa.get_analysis()
    listst = sorted(t.items()) # sorted by key, return a list of tuples
    x, y = zip(*listst) # unpack a list of pairs into two tuples
    dd = {'data':x, '%s'%params['model_params']['model_n']:np.cumsum(y)}
    df = pd.DataFrame(dd).set_index('data')
    return df

# import strategy
from backtest import optimizer, Model

def run_backtest(stock, params, daily_return, tickers,**kwargs):
    # launch cerebro
    print(params['model_params']['model_n'])
    cerebro = bt.Cerebro()

    # set initial cash
    cerebro.broker.set_cash(100000)
    print('Starting Portfolio Value: %.2f' % cerebro.broker.getvalue())


    # add returns_data to cerebro datafeed
    for a in tickers:
        cerebro.adddata(bt.feeds.PandasData(dataname=getStock(stock, a), name=a))

    #add analyzers
    if daily_return == True:
        cerebro.addanalyzer(bt.analyzers.TimeReturn, timeframe=bt.TimeFrame.Days, _name='PortfolioReturns')
    else:
        cerebro.addanalyzer(bt.analyzers.TimeReturn, timeframe=bt.TimeFrame.Months, _name='PortfolioReturns')
    cerebro.addanalyzer(bt.analyzers.SharpeRatio, riskfreerate=0.0, timeframe=bt.TimeFrame.Months, _name='mysharpe')
    cerebro.addanalyzer(bt.analyzers.Returns, _name='myreturn')
    cerebro.addanalyzer(bt.analyzers.DrawDown, _name='mydrawdown')
    cerebro.addanalyzer(bt.analyzers.AnnualReturn, _name='_AnnualReturn')

    # add strategy
    cerebro.addstrategy(optimizer, **kwargs)


    # run strategy
    thestrats = cerebro.run()

    thestrat = thestrats[0]


    # #analysis module
    # stats =  {'Max_Drawdown':thestrat.analyzers.getbyname('mydrawdown').get_analysis()['max']['drawdown'],
    #         'Value': cerebro.broker.getvalue()
    #         }

    #analysis module
    stats =  {'Max_Drawdown':thestrat.analyzers.getbyname('mydrawdown').get_analysis()['max']['drawdown'],
            'CAGR':thestrat.analyzers.getbyname('myreturn').get_analysis()['rnorm100'],
            'Sharpe_Ratio':thestrat.analyzers.getbyname('mysharpe').get_analysis()['sharperatio'],
            'Value': cerebro.broker.getvalue()
            }
    annual_return = thestrat.analyzers._AnnualReturn.get_analysis()
    portfolio_return = thestrat.analyzers.PortfolioReturns.get_analysis()
    df_cum_returns = getReturnAsDataFrame(params, thestrat.analyzers.getbyname('PortfolioReturns'))


# print output
#     for key, value in stats.items():
#         print("--------------- %s -----------------" %key)
#         print(value)
#     print("--------------- Annual Returns -----------------")
#     print(dict(annual_return))


    # df_cum_returns.to_csv('cum_returns_data/model_returns'+'_%s'%params['model_params']['model_n'] +'.csv')
    df_returns = pd.DataFrame([portfolio_return.values()], columns=portfolio_return.keys(),
                              index=[params['model_params']['model_n']]).T
    # df_returns.to_csv('returns_data/model_returns' + '_%s' % params['model_params']['model_n'] + '.csv')
    df_stats = pd.DataFrame(stats.values(), index = stats.keys(), columns=[params['model_params']['model_n']])
    # df_stats.to_csv('stats_data/model_stats' + '_%s' % params['model_params']['model_n'] + '.csv')


    print('Final Portfolio Value: %.2f' % cerebro.broker.getvalue())
    return df_returns, df_stats, df_cum_returns, thestrats[0].weights_vector

#create list of models
# Tests for total portfolio were conducted for three sample periods:
# 2021.1.1 - 2021.12.31, 2022.1.1-2022.12.12, 2021.1.1 - 2023.1.1.
# Tests for dual portfolio were conducted for three sample periods:
# 2020.1.1 - 2021.12.31, 2022.1.1-2022.12.12, 2020.1.1 - 2023.1.1.
    # models = ['MV', 'Sharpe_Ratio','HRP', 'risk_parity', 'cvar','cdar','random','IVP', 'eigen','equal']
    # get returns_data
    # tickers = ['BTC-USD' ,'ETH-USD','USDC-USD','USDT-USD','DAI-USD']
    # tickers = ['USDC-USD','USDT-USD','DAI-USD']


class Full_Pipeline():
    def __init__(self, models,start,end,tickers, warmup, rebalance, predicted_periods, printlog = False, EMA = True):
        self.models = models
        self.tickers = tickers
        self.warmup = warmup
        self.rebalance = rebalance
        self.printlog = printlog
        self.start = start - timedelta(days=warmup)
        self.end = end
        self.EMA = EMA
        self.predicted_periods = predicted_periods

    def run_model(self, save_res, plot_save, daily_return, plot_show = True):
        stock = pdr.get_data_yahoo(self.tickers,
                        start= self.start,
                        end= self.end, interval = '1d')
        df_returns_all = pd.DataFrame()
        df_cum_returns_all = pd.DataFrame()
        df_stats_all = pd.DataFrame()
        all_weights = []
        for model in tqdm(self.models):
            params = {'Warmup': self.warmup,
                      'RebalanceDay': self.rebalance,
                      'printlog': self.printlog,
                      'model': Model(),
                      'tickers' : self.tickers,
                      'model_params':
                          {'model_n': '%s'%model,
                           'isCleanWeight': False,
                           'EMA': self.EMA,
                           'predicted_periods' : self.predicted_periods}
                      }
            df_returns, df_stats, df_cum_returns,weights_vector = run_backtest(stock,params,daily_return, **params)
            df_cum_returns_all = pd.concat([df_cum_returns_all, df_cum_returns], axis=1)
            if daily_return == True:
                df_cum_returns_all = df_cum_returns_all[self.warmup:]
            else:
                df_cum_returns_all = df_cum_returns_all[int(self.warmup/30.44):]
            df_returns_all = pd.concat([df_returns_all, df_returns], axis=1)
            df_stats_all = pd.concat([df_stats_all, df_stats], axis=1)
            all_weights.append(weights_vector)
            warmup = params['Warmup']
            rebalance_day = params['RebalanceDay']




        if save_res == True:
            #Save summary to csv
            df_returns_all.to_csv('returns_data/model_returns_all_warmup_%d_rebal_%d.csv'%(int(warmup), int(rebalance_day)))
            df_cum_returns_all.to_csv('cum_returns_data/model_cum_warmup_%d_rebal_%d.csv'%(int(warmup), int(rebalance_day)))
            df_stats_all.to_csv('stats_data/model_stats_all_warmup_%d_rebal_%d.csv'%(int(warmup), int(rebalance_day)))

        if plot_save == True:
            #Plot and save
            plotReturns(df_cum_returns_all, warmup, rebalance_day, save=True)
            for i in df_stats_all.T:
                plotStatsValues(df_stats_all.T[i],'%s'%i,warmup, rebalance_day,save=True)


        if plot_show == True:
            #Or just show
            plotReturns(df_cum_returns_all, warmup, rebalance_day, show=True)
            for i in df_stats_all.T:
                plotStatsValues(df_stats_all.T[i],'%s'%i,warmup, rebalance_day, show=True)

        st.subheader('Cumulative Returns')
        st.dataframe(df_cum_returns_all)
        st.subheader('Summary Statistics')
        st.dataframe(df_stats_all)
        all_weights = pd.DataFrame(all_weights, index=self.models).T
        st.subheader('Weights over time')
        st.dataframe(all_weights)

