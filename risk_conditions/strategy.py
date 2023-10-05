import yfinance as yf
from pandas_datareader import data as pdr
import backtrader as bt
import numpy as np
import pandas as pd
import plotly.express as px
# import matplotlib.pyplot as plt
from datetime import datetime
yf.pdr_override()

idx = pd.IndexSlice
def getStock(df, s):
    c = df.loc[:,idx[:,s]]
    c.columns = c.columns.droplevel(1)
    c = pd.DataFrame(c.to_records()).set_index('Datetime').dropna()
    if c.isna().any().any():
        raise ValueError("Array contains NaN values which means that there is no available data for some asset during some part of chosen period"
                         " also you cannot mix shares with crypto assets as shares are not traded on weekends")
    return c
def getReturnAsDataFrame(pfa):
    t = pfa.get_analysis()
    listst = sorted(t.items()) # sorted by key, return a list of tuples
    x, y = zip(*listst) # unpack a list of pairs into two tuples
    dd = {'data':x, 'returns' :np.cumsum(y)}
    df = pd.DataFrame(dd).set_index('data')
    return df

class risk_manager(bt.Strategy):
    def __init__(self):
        self.high = 0

    def next(self):
        self.high +=1
        print(self.high)

class optimizer(bt.SignalStrategy):

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


    def getDailyPrice(self):
        p = {}
        for i in range(self.nDatas):
            p[str(i)] = np.array(self.datas[i].get(size=1))
        date_obj= self.data.datetime.date(-1)
        return pd.DataFrame(p, index = [date_obj])


    def getModelDataFrame(self):
        p = {}
        for i in range(self.nDatas):
            p[str(i)] = np.array(self.datas[i].get(size=self.window))
        return pd.DataFrame(p)

   #initialize strategy
    def __init__(self):
        self.counter = 0
        self.update_counter = 0
        self.isFirst = True
        self.old_pct = []
        self.new_pct = []
        self.nDatas = len(self.datas)
        self.verbose = 0  # 0
        self.weights_vector = []
        self.drawdown_threshold = 0.1
        self.max_portfolio_value = 0
        self.cooldown = 0
        self.trigger = False
        self.window = 5
        self.returns = bt.analyzers.Returns()
        self.daily_data = pd.DataFrame()
        self.RSI = bt.indicators.RelativeStrengthIndex()
        pass

    #create conditions for strategy
    def next(self):
        current_date = self.data.datetime.date(0)
        last_date = self.data.datetime.date(-1)

        # getting daily data
        # if current_date != last_date:
        #     m_input = self.getDailyPrice()
        #     self.daily_data = pd.concat([self.daily_data, m_input],ignore_index=False)
        # last_n_days = self.daily_data.tail(self.window)
        # mean_volatility = None
        # if len(last_n_days) == self.window:
        #     returns = last_n_days.pct_change()
        #     volatility = returns.std()
        #     mean_volatility = volatility.mean()

        #drawdown condition
        portfolio_value = self.broker.get_value()
        if portfolio_value > self.max_portfolio_value:
            self.max_portfolio_value = portfolio_value
        drawdown = (portfolio_value - self.max_portfolio_value) / self.max_portfolio_value

        # if mean_volatility is not None:
        if self.RSI < 16 and self.isFirst == False and self.trigger == True:
            self.isFirst = True


        if self.isFirst == True:
            self.new_pct = np.array([0.5, 0.5])
            self.old_pct = np.zeros(len(self.new_pct))

            if self.verbose == True:
                print('Current portfolio value %s' % self.broker.getvalue())
                print('Weights for current epoch %s' % self.new_pct)
            self.weights_vector.append(list(self.new_pct))

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
                self.sell(self.datas[i], p)
            for i, p in to_buy:
                self.buy(self.datas[i], p)
            self.old_pct = self.new_pct
            self.isFirst = False
            self.trigger = False


        elif self.RSI > 70:
            self.new_pct = np.array([0, 0])
            self.old_pct = np.zeros(len(self.new_pct))

            if self.verbose == True:
                print('Current portfolio value %s' % self.broker.getvalue())
                print('Weights for current epoch %s' % self.new_pct)
            self.weights_vector.append(list(self.new_pct))

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
                self.sell(self.datas[i], p)
            for i, p in to_buy:
                self.buy(self.datas[i], p)

            self.old_pct = self.new_pct
            self.max_portfolio_value = 0
            self.trigger = True


# strategy setup
start = datetime(2022, 1, 1)
end = datetime(2023, 1, 1)
tickers = ['BTC-USD', 'ETH-USD']
stock = yf.download(tickers, start=start, end=end, interval="1h")
# stock = pdr.get_data_yahoo(tickers,
#                         start=start,
#                         end=end, interval = '1ho')


#START OF THE STRATEGY
cerebro = bt.Cerebro()
cerebro.broker.set_cash(100000)

print('Starting Portfolio Value: %.2f' % cerebro.broker.getvalue())

# add data
for ticker in tickers:
    cerebro.adddata(bt.feeds.PandasData(dataname=getStock(stock, ticker), name=ticker))

# add analyzers
cerebro.addanalyzer(bt.analyzers.TimeReturn, timeframe=bt.TimeFrame.Days, _name='PortfolioReturns')
cerebro.addanalyzer(bt.analyzers.SharpeRatio, riskfreerate=0.0, timeframe=bt.TimeFrame.Months, _name='mysharpe')
cerebro.addanalyzer(bt.analyzers.Returns, _name='myreturn')
cerebro.addanalyzer(bt.analyzers.DrawDown, _name='mydrawdown')
cerebro.addanalyzer(bt.analyzers.AnnualReturn, _name='_AnnualReturn')

# add strategy
cerebro.addstrategy(optimizer)

# run strategy
thestrats = cerebro.run()
thestrat = thestrats[0]

#analysis module
stats =  {'Max_Drawdown':thestrat.analyzers.getbyname('mydrawdown').get_analysis()['max']['drawdown'],
        'CAGR':thestrat.analyzers.getbyname('myreturn').get_analysis()['rnorm100'],
        'Sharpe_Ratio':thestrat.analyzers.getbyname('mysharpe').get_analysis()['sharperatio'],
        'Value': cerebro.broker.getvalue()
        }
annual_return = thestrat.analyzers._AnnualReturn.get_analysis()
portfolio_return = thestrat.analyzers.PortfolioReturns.get_analysis()
df_cum_returns = getReturnAsDataFrame(thestrat.analyzers.getbyname('PortfolioReturns'))

print('Final Portfolio Value: %.2f' % cerebro.broker.getvalue())

cerebro.plot()

fig = px.line(df_cum_returns)
fig.update_layout(coloraxis=dict(colorscale='Viridis'))
fig.show()

df_stats = pd.DataFrame(stats.values(), index = stats.keys())
print(df_stats)



