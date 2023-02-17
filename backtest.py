import yfinance as yf
from pandas_datareader import data as pdr
from datetime import datetime
import numpy as np
import pandas as pd
import backtrader as bt
from pypfopt import HRPOpt, risk_models, expected_returns

yf.pdr_override()

#get data
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

class Model:
    def __init__(self):
        self.data = None
        self.isFirst = True

    def get_allocations(self, data, **param):

        data = pd.DataFrame(data)

        model_n = param['model_n']
        isCleanWeight = param['isCleanWeight']

        mu = expected_returns.mean_historical_return(self.data).fillna(0.0).values

        S = risk_models.sample_cov(self.data).fillna(0.0).values

        self.data = data.copy()

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
        ('model_params',
         { 'model_n':'HRP',
        'isCleanWeight':False}
         ),
        ('printlog', 1),
        ('model', Model())
    )

    def getPosDifference(self, cash, alloc, new_price, cur_pos):
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
        self.verbose = self.params.printlog  # 0
        self.model = self.params.model
        self.model_params = self.params.model_params
        pass

    def next(self):
        if self.counter < self.DataCounter:
            self.counter += 1
            return

        m_input = self.getModelDataFrame()
        if self.update_counter == 0:
            self.new_pct = np.round(self.model.get_allocations(m_input.values, **self.model_params), 2)
            self.old_pct = np.zeros(len(self.new_pct))

            if self.verbose == 1:
                print('Current portfolio value %s' % self.broker.getvalue())
                print('Weights for current epoch %s' % self.new_pct)

            cash = self.broker.get_cash()
            new_price = self.getCurrentClosePrice()
            cur_pos = self.getPosSize()
            upd_pos = self.getPosDifference(cash, self.new_pct, new_price, cur_pos)
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

thestrats = cerebro.run()
thestrat = thestrats[0]

cerebro.plot()

print('Final Portfolio Value: %.2f' % cerebro.broker.getvalue())




