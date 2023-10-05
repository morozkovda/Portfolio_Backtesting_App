import yfinance as yf
import numpy as np
import pandas as pd
import backtrader as bt
from pypfopt import EfficientFrontier, EfficientCVaR, EfficientCDaR, CLA,  HRPOpt, risk_models, expected_returns
from scipy.special import softmax
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from backtrader import Analyzer, TimeFrame
import streamlit as st
# import tensorflow as tf
# from keras.layers import LSTM, Flatten, Dense
# from keras.models import Sequential
# import keras.backend as K
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_regression
import statsmodels.api as sm
# from keras.optimizers import SGD
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
# from keras import metrics
from statsmodels.tsa.statespace.sarimax import SARIMAX


yf.pdr_override()

# create class that will contain all the models
class Model:
    def __init__(self):
        self.data = None
        self.isFirst = True
        self.predicted_periods = None
        self.tensored_data = None


    # you can create custom optimization obejectives like this
    def deviation_risk_parity(self, w, cov_matrix):
        diff = w * np.dot(cov_matrix, w) - (w * np.dot(cov_matrix, w)).reshape(-1, 1)
        return (diff ** 2).sum().sum()

    def build_MAXSH_model(self, input_shapes, outputs):

        @tf.function
        def sharpe_loss(_, y_pred):
            data = tf.divide(self.tensored_data, self.tensored_data[0])
            portfolio_values = tf.reduce_sum(tf.multiply(data, y_pred), axis=1)

            portfolio_returns = (portfolio_values[1:] - portfolio_values[:-1]) / portfolio_values[:-1]

            sharpe = K.mean(portfolio_returns) / K.std(portfolio_returns)

            # since we want to maximize Sharpe, while gradient descent minimizes the loss,
            #   we can negate Sharpe (the min of a negated function is its max)
            return -sharpe

        model = Sequential([
            LSTM(64, input_shape=input_shapes),
            Flatten(),
            Dense(outputs, activation='softmax')
        ])

        model.compile(loss=sharpe_loss, optimizer=tf.keras.optimizers.Adam(learning_rate=0.1))
        return model

    def predict_return_LSTM(self, data, predicted_periods):
        return_period = 1
        Y = np.log(data).diff(return_period).shift(-return_period)
        X = pd.concat([np.log(data).diff(i) for i in
                                [return_period, return_period * 3, return_period * 6, return_period * 12]], axis=1).dropna()
        data = pd.concat([Y, X], axis=1).dropna().iloc[::return_period, :]

        scaler = MinMaxScaler()
        data = scaler.fit_transform(data.values.reshape(-1, 1))
        train_data, valid_data = train_test_split(data, test_size=0.2, shuffle=False)

        window_size = len(self.data)
        train_X, train_y = [], []
        for i in range(window_size, len(train_data)):
            train_X.append(train_data[i - window_size:i, 0])
            train_y.append(train_data[i, 0])
        train_X, train_y = np.array(train_X), np.array(train_y)
        train_X = np.reshape(train_X, (train_X.shape[0], train_X.shape[1], 1))


        model = Sequential()
        model.add(LSTM(units=64, return_sequences=True, input_shape=(train_X.shape[1], 1)))
        model.add(LSTM(units=32, return_sequences=False))
        model.add(Dense(units=1))

        model.compile(optimizer='adam', loss='mean_squared_error')

        model.fit(train_X, train_y, epochs=50, batch_size=32, validation_split=0.2, verbose=2)
        # plt.plot(model.history.history['loss'], label='Training Loss')
        # plt.plot(model.history.history['val_loss'], label='Validation Loss')
        # plt.title('Learning Curve')
        # plt.xlabel('Epoch')
        # plt.ylabel('Loss')
        # plt.legend()
        # st.pyplot()

        inputs = data[-window_size:, :]
        inputs = np.reshape(inputs, (1, window_size, 1))
        predictions = []
        for i in range(predicted_periods):
            prediction = model.predict(inputs)
            predictions.append(prediction)
            inputs = np.append(inputs[:, 1:, :], prediction.reshape(1, 1, 1), axis=1)

        predictions = np.array(predictions)
        predictions = scaler.inverse_transform(predictions.reshape(-1, 1))
        return(predictions)

    def predict_return_ARIMA(self, data, predicted_periods):
        endog = data
        train_size = int(len(endog) * 0.8)
        train_endog, test_endog = endog[:train_size], endog[train_size:]
        model = SARIMAX(train_endog, order=(1, 1, 1), disp=0)
        model_fit = model.fit()
        predictions = model_fit.forecast(steps=predicted_periods)
        print(predicted_periods)
        return predictions


    # create function to get weights from model
    def get_allocations(self, data, **param):

        data = pd.DataFrame(data)
        self.data = data.copy()
        model_n = param['model_n']
        isCleanWeight = param['isCleanWeight']
        ema = param['EMA']
        self.predicted_periods = param['predicted_periods']

        mu = expected_returns.mean_historical_return(self.data).fillna(0.0).values
        if ema == True:
            mu = expected_returns.ema_historical_return(self.data).fillna(0.0).values

        S = risk_models.sample_cov(self.data).fillna(0.0).values
        if ema == True:
         S = risk_models.exp_cov(self.data).fillna(0.0).values

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

        elif model_n == 'cvar':
            ef = EfficientCVaR(mu, S)
            weights = ef.min_cvar()
            cleaned_weights = ef.clean_weights()
            if isCleanWeight == True:
                return pd.DataFrame(cleaned_weights, index=[0]).to_numpy()[0]
            return pd.DataFrame(weights, index=[0]).to_numpy()[0]

        elif model_n == 'cdar':
            ef = EfficientCDaR(mu, S)
            weights = ef.min_cdar()
            cleaned_weights = ef.clean_weights()
            if isCleanWeight == True:
                return pd.DataFrame(cleaned_weights, index=[0]).to_numpy()[0]
            return pd.DataFrame(weights, index=[0]).to_numpy()[0]

        elif model_n == 'eigen':
            rets = self.data.pct_change()
            scaler = MinMaxScaler()
            rets_scal = pd.DataFrame(scaler.fit_transform(rets), columns=rets.columns, index=rets.index)
            rets_scal.dropna(how='any', inplace=True)
            pca = PCA()
            pca.fit(rets_scal)
            weights = pd.DataFrame()

            for i in range(len(pca.components_)):
                weights["weights_{}".format(i)] = pca.components_[i] / sum(pca.components_[i])
            weights = weights.values.T
            return weights[0]

        elif model_n == 'eigen_modified':
            rets = self.data.pct_change()
            scaler = MinMaxScaler()
            rets_scal = pd.DataFrame(scaler.fit_transform(rets), columns=rets.columns, index=rets.index)
            rets_scal.dropna(how='any', inplace=True)
            pca = PCA()
            pca.fit(rets_scal)
            weights = pd.DataFrame()

            for i in range(len(pca.components_)):
                weights["weights_{}".format(i)] = pca.components_[i] / sum(pca.components_[i])
            weights = weights.values.T
            weights = weights[0]
            weights[weights < 0] = 0
            sorted_indices = np.argsort(weights)[::-1]
            weights = weights[sorted_indices]
            weights = weights / np.sum(weights)
            weights = weights[np.argsort(sorted_indices)]
            return weights

        elif model_n == 'eigen_modified_2':
            rets = self.data.pct_change()
            scaler = MinMaxScaler()
            rets_scal = pd.DataFrame(scaler.fit_transform(rets), columns=rets.columns, index=rets.index)
            rets_scal.dropna(how='any', inplace=True)
            pca = PCA()
            pca.fit(rets_scal)
            weights = pd.DataFrame()

            for i in range(len(pca.components_)):
                weights["weights_{}".format(i)] = pca.components_[i] / sum(pca.components_[i])
            weights = weights.values.T
            weights = weights[1]
            weights[weights < 0] = 0
            sorted_indices = np.argsort(weights)[::-1]
            weights = weights[sorted_indices]
            weights = weights / np.sum(weights)
            weights = weights[np.argsort(sorted_indices)]
            return weights

        elif model_n == 'IVP':
            weights = [round((1 / i) / sum(1 / np.diagonal(S)), 3) for i in np.diagonal(S)]
            return weights

        elif model_n == 'random':
            a = np.random.uniform(size=[self.data.shape[1]])
            return softmax(a)

        elif model_n == 'MAXSH_LSTM':
            def scale_0_1(data):
                return (data - np.min(data, axis=0)) / (np.max(data, axis=0) - np.min(data, axis=0) + 0.000000001)
            local_data = self.data.copy()
            data_w_ret = np.concatenate([scale_0_1(local_data[1:]), local_data.pct_change().dropna()], axis=1)
            self.tensored_data = tf.cast(tf.constant(local_data), float)
            fit_predict_data = data_w_ret[np.newaxis, :]
            callback_early_stop = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=20)
            model = self.build_MAXSH_model(data_w_ret.shape, local_data.shape[1])
            model.fit(fit_predict_data, np.zeros((1, local_data.shape[1])), epochs=100,
                           shuffle=False, callbacks=[callback_early_stop], verbose=0)
            weights = model.predict(fit_predict_data)[0]
            return np.round(weights,2)

        elif model_n == 'eigen_modified_ARIMA':
            local_data = self.data.copy()
            rets = local_data.pct_change()
            rets.dropna(how='any', inplace=True)
            predictions_df = pd.DataFrame()
            for i in rets.columns:
                predictions = self.predict_return_ARIMA(rets[i], self.predicted_periods)
                predictions_df[i] = predictions
            rets = pd.concat([rets,predictions_df], axis=0,ignore_index=True)
            rets.reset_index()
            scaler = MinMaxScaler()
            rets_scal = pd.DataFrame(scaler.fit_transform(rets), columns=rets.columns, index=rets.index)
            rets_scal.dropna(how='any', inplace=True)
            pca = PCA()
            pca.fit(rets_scal)
            weights = pd.DataFrame()

            for i in range(len(pca.components_)):
                weights["weights_{}".format(i)] = pca.components_[i] / sum(pca.components_[i])
            weights = weights.values.T
            weights = weights[0]
            weights[weights < 0] = 0
            sorted_indices = np.argsort(weights)[::-1]
            weights = weights[sorted_indices]
            weights = weights / np.sum(weights)
            weights = weights[np.argsort(sorted_indices)]
            return weights

        elif model_n == 'eigen_modified_LSTM':
            local_data = self.data.copy()
            rets = np.log(local_data).diff(1)
            predictions_df = pd.DataFrame()
            for i in rets.columns:
                predictions = self.predict_return_LSTM(local_data[i], self.predicted_periods)
                predictions_df[i] = predictions.flatten()
            rets = pd.concat([rets,predictions_df], axis=0,ignore_index=True)
            rets.reset_index()
            scaler = MinMaxScaler()
            rets_scal = pd.DataFrame(scaler.fit_transform(rets), columns=rets.columns, index=rets.index)
            rets_scal.dropna(how='any', inplace=True)
            pca = PCA()
            pca.fit(rets_scal)
            weights = pd.DataFrame()
            for i in range(len(pca.components_)):
                weights["weights_{}".format(i)] = pca.components_[i] / sum(pca.components_[i])
            weights = weights.values.T
            weights = weights[0]
            weights[weights < 0] = 0
            sorted_indices = np.argsort(weights)[::-1]
            weights = weights[sorted_indices]
            weights = weights / np.sum(weights)
            weights = weights[np.argsort(sorted_indices)]
            return weights



        elif model_n == 'equal':
            a = np.array([1 / self.data.shape[1]] * self.data.shape[1])
            return a


        return []



# create strategy for backtrader
class optimizer(bt.SignalStrategy):
    # set default params
    params = (
        ('Warmup', 125),
        ('RebalanceDay', 22),
        ('printlog', False),
        ('model', Model()),
         ('model_params',
         { 'model_n':'HRP',
        'isCleanWeight': False,
           'EMA': True,
           'predicted_periods' : None}
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
            p[str(i)] = np.array(self.datas[i].get(size=self.Warmup))
        return pd.DataFrame(p)

   #initialize strategy
    def __init__(self):
        self.counter = 0
        self.update_counter = 0
        self.isFirst = True
        self.old_pct = []
        self.new_pct = []
        self.nDatas = len(self.datas)
        self.Warmup = self.params.Warmup  # 125
        self.RebalanceDay = self.params.RebalanceDay  # 22
        self.verbose = self.params.printlog  # 0
        self.model = self.params.model
        self.model_params = self.params.model_params
        self.weights_vector = []
        pass

    #create conditions for strategy
    def next(self):

        if self.counter < self.Warmup:
            self.counter += 1
            return

        m_input = self.getModelDataFrame()
        if self.update_counter == 0:
            self.new_pct = np.round(self.model.get_allocations(m_input.values, **self.model_params), 2)
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
        self.update_counter += 1
        if self.update_counter == self.RebalanceDay:
            self.update_counter = 0


