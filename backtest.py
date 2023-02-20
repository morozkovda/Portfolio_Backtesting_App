import yfinance as yf
import numpy as np
import pandas as pd
import backtrader as bt
from pypfopt import EfficientFrontier, HRPOpt, risk_models, expected_returns
from scipy.special import softmax
yf.pdr_override()

# create class that will contain all the models
class Model:
    def __init__(self):
        self.data = None
        self.isFirst = True

    # create function to get weights from model
    def deviation_risk_parity(self, w, cov_matrix):
        diff = w * np.dot(cov_matrix, w) - (w * np.dot(cov_matrix, w)).reshape(-1, 1)
        return (diff ** 2).sum().sum()

    def get_allocations(self, data, **param):

        data = pd.DataFrame(data)
        self.data = data.copy()

        model_n = param['model_n']
        isCleanWeight = param['isCleanWeight']

        mu = expected_returns.mean_historical_return(self.data).fillna(0.0).values

        S = risk_models.sample_cov(self.data).fillna(0.0).values


        #here you can add your models

        if model_n == 'MV':
            ef = EfficientFrontier(mu, S)
            weights = ef.min_volatility()
            cleaned_weights = ef.clean_weights()
            if isCleanWeight == True:
                return pd.DataFrame(cleaned_weights, index=[0]).to_numpy()[0]
            return pd.DataFrame(weights, index=[0]).to_numpy()[0]

        elif model_n == 'HRP':
            hrp = HRPOpt(self.data.pct_change(), S)
            weights = hrp.optimize()
            cleaned_weights = hrp.clean_weights()
            if isCleanWeight == True:
                return pd.DataFrame(cleaned_weights, index=[0]).to_numpy()[0]
            return pd.DataFrame(weights, index=[0]).to_numpy()[0]

        elif model_n == 'risk_parity':
            ef = EfficientFrontier(mu, S)
            weights = ef.nonconvex_objective(self.deviation_risk_parity, ef.cov_matrix)
            cleaned_weights = ef.clean_weights()
            if isCleanWeight == True:
                return pd.DataFrame(cleaned_weights, index=[0]).to_numpy()[0]
            return pd.DataFrame(weights, index=[0]).to_numpy()[0]

        elif model_n == 'random':
            a = np.random.uniform(size=[self.data.shape[1]])
            return softmax(a)
        elif model_n == 'equal':
            a = np.array([1 / self.data.shape[1]] * self.data.shape[1])
            return a


        return []


# create strategy for backtrader
class optimizer(bt.SignalStrategy):
    # set default params
    params = (
        ('DataCounter', 125),
        ('RebalanceDay', 22),
        ('printlog', 0),
        ('model', Model()),
         ('model_params',
         { 'model_n':'HRP',
        'isCleanWeight':False}
          )
    )

    # create function to calculate difference from previous and current positiom
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

    # get position size
    def getPosSize(self):
        p = []
        for i in range(self.nDatas):
            p.append(self.getposition(data=self.datas[i]).size)
        return np.array(p)

    # get close prices
    def getCurrentClosePrice(self):
        p = []
        for i in range(self.nDatas):
            p.append(self.datas[i].close[0])
        return np.array(p)

    #get open prices
    def getPosOpenPrice(self):
        p = []
        for i in range(self.nDatas):
            p.append(self.getposition(data=self.datas[i]).price)
        return np.array(p)

    # get dataframe from bt.feeds for the model
    def getModelDataFrame(self):
        p = {}
        for i in range(self.nDatas):
            p[str(i)] = np.array(self.datas[i].get(size=self.DataCounter))
        return pd.DataFrame(p)

   #initialize strategy
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

    #create conditions for strategy
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






