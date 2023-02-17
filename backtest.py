import yfinance as yf
from pandas_datareader import data as pdr
from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import copy
import backtrader as bt
import backtrader.feeds as btfeeds
yf.pdr_override()

#get data
tickers = ['SPY','META','AAPL']
stock = pdr.get_data_yahoo(tickers,
                start= datetime(2015, 1, 1),
                end= datetime(2023, 2, 10), interval = '1d')
stock.head()

very_small_float = 1.0
idx = pd.IndexSlice
def getStock(df, s):
    c = df.loc[:,idx[:,s]]
    c.columns = c.columns.droplevel(1)
    c = pd.DataFrame(c.to_records()).set_index('Date')
    c = c.fillna(very_small_float)
    return c

class Model:
    def __init__(self):
        self.data = None
        self.isFirst = True


    def resetModel(self):
        self.data = None
        self.isFirst = True
        pass

    # a nonconvex objective from  Kolm et al (2014)
    def deviation_risk_parity(self, w, cov_matrix):
        diff = w * np.dot(cov_matrix, w) - (w * np.dot(cov_matrix, w)).reshape(-1, 1)
        return (diff ** 2).sum().sum()

    # def get_allocations(self, data, model_n = 'MeanVariance', isEmsRet = False, isExpCov = False, isCleanWeight = False, upd_period = 0):
    def get_allocations(self, data, **param):

        data = pd.DataFrame(data)

        model_n = param['model_n']
        isEmsRet = param['isEmsRet']
        isExpCov = param['isExpCov']
        isCleanWeight = param['isCleanWeight']
        upd_period = param['upd_period']
        isDataCollecting = param['collectData']

        # data = data.fillna(0.0)
        if (self.isFirst == True) or (upd_period == 0) or (isDataCollecting == False):
            self.isFirst = False
            self.data = data.copy()
        else:
            self.data = pd.concat([self.data, data.iloc[-upd_period:]])

        # print(data.iloc[-upd_period:])

        if model_n == 'HRP':
            hrp = HRPOpt(self.data.pct_change(), S)
            weights = hrp.optimize()
            cleaned_weights = hrp.clean_weights()
            if isCleanWeight == True:
                return pd.DataFrame(cleaned_weights, index=[0]).to_numpy()[0]
            return pd.DataFrame(weights, index=[0]).to_numpy()[0]

        return []


class optimizer(bt.SignalStrategy):
    params = (
        ('DataCounter', 125),
        ('RebalanceDay', 22),
        ('reserveCash', 1000.0),
    )

    def getPosDiffiretce(self, cash, alloc, new_price, cur_pos):
        pos_cash = new_price * cur_pos
        # print('pos_cash',pos_cash)
        all_cash = cash + np.sum(pos_cash)
        # print('all_cash',all_cash)
        cash_alloc = alloc * all_cash
        # print('cash_alloc',cash_alloc)
        new_pos = (cash_alloc / new_price).astype(int)
        # print('new_pos',new_pos)
        diff_pos = cur_pos - new_pos
        return diff_pos * (-1)

    def getPosSize(self):
        p = []
        for i in range(self.nDatas):
            p.append(self.getposition(data=self.datas[i]).size)
        return np.array(p)

    def getCurrentClosePrice(self):
        p = []
        for i in range(self.nDatas):
            p.append(self.datas[i].close[0])
        return np.array(p)

    def getPosOpenPrice(self):
        p = []
        for i in range(self.nDatas):
            p.append(self.getposition(data=self.datas[i]).price)
        return np.array(p)

    def getModelDataFrame(self):
        p = {}
        for i in range(self.nDatas):
            p[str(i)] = np.array(self.datas[i].get(size=self.DataCounter))
        return pd.DataFrame(p)

    def __init__(self):
        self.counter = 0
        self.update_counter = 0
        self.isFirst = True
        self.old_pct = []
        self.new_pct = []
        self.nDatas = len(self.datas)
        self.DataCounter = self.params.DataCounter  # 125
        self.RebalanceDay = self.params.RebalanceDay  # 22
        self.reserveCash = self.params.reserveCash  # 1000.0

        self.verbose = self.params.printlog  # 0
        pass

    def next(self):
        if self.counter < self.DataCounter:
            self.counter += 1
            return

        m_input = self.getModelDataFrame()
        if self.update_counter == 0:

            self.new_pct = np.round(self.model.get_allocations(m_input.values, **self.model_params), 2)

            if self.isFirst == True:
                self.isFirst = False
                self.old_pct = np.zeros(len(self.new_pct))
                #
                # if np.array_equal(self.new_pct, self.old_pct) == False:
                #     if self.verbose > 1:
                #         print('size', self.getPosSize().tolist())
                #         print('price', self.getPosOpenPrice().tolist())
                #     if self.verbose > 0:
                #         print("rebalance new percent.",self.new_pct)

                cash = self.broker.get_cash()

                new_price = self.getCurrentClosePrice()
                cur_pos = self.getPosSize()
                upd_pos = self.getPosDiffiretce(cash, self.new_pct, new_price, cur_pos)
                to_sell = []
                to_buy = []
                for i, p in enumerate(upd_pos):
                    if (p < 0):
                        to_sell.append((i, p))
                    elif (p > 0):
                        to_buy.append((i, p))

                for i, p in to_sell:
                    if self.verbose > 0:
                        print('sell', i, p)
                    self.sell(self.datas[i], p)
                for i, p in to_buy:
                    if self.verbose > 0:
                        print('buy', i, p)
                    self.buy(self.datas[i], p)

                self.old_pct = self.new_pct

        self.update_counter += 1
        if self.update_counter == self.RebalanceDay:
            self.update_counter = 0





cerebro = bt.Cerebro()

cerebro.broker.set_cash(10000)
print('Starting Portfolio Value: %.2f' % cerebro.broker.getvalue())


for a in tickers:
    cerebro.adddata(bt.feeds.PandasData(dataname=getStock(stock, a), name=a))

cerebro.addstrategy(optimizer)

cerebro.run()

print('Final Portfolio Value: %.2f' % cerebro.broker.getvalue())


