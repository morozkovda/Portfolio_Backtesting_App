from backtest_setup import Full_Pipeline
import streamlit as st
from datetime import datetime, timedelta

if __name__ == '__main__':
    st.set_page_config(page_title='Portfolio Optimization App')
    app = st.container()
    with app:
        models = st.sidebar.multiselect('Select options:', ['MV','HRP', 'risk_parity', 'cvar','cdar','random','IVP',
                                                            'eigen','eigen_modified','eigen_modified_2',
                                                            'eigen_modified_LSTM', 'eigen_modified_ARIMA',
                                                            'MAXSH_LSTM','equal'])
        default_start = datetime(2022, 1, 1)
        default_end = datetime(2023, 1, 1)
        start = st.sidebar.date_input("Start Date",default_start)
        end = st.sidebar.date_input("End Date", default_end)
        warmup = st.sidebar.slider("Warmup Days", min_value=0, max_value=365, value=50)
        rebalance = st.sidebar.slider("Rebalance Days", min_value=0, max_value=365, value=30)
        predicted_periods = st.sidebar.slider("Periods to predict", min_value=0, max_value=20, value=0)
        options = ['BTC-USD' ,'ETH-USD','USDC-USD','USDT-USD','DAI-USD', 'BNB-USD', 'SPY','AAPL', 'META']
        user_input = st.sidebar.text_input("Enter your custom ticker using Yahoo Finance (e.g. SPY) :")
        if "user_inputs" not in st.session_state:
            st.session_state.user_inputs = []
        add_button = st.sidebar.button("Add")
        if add_button:
            st.session_state.user_inputs.append(user_input)
        for i in st.session_state.user_inputs:
            options.append(i)
        tickers = st.sidebar.multiselect("Tickers: ", options, key="selected")
        daily_return = st.sidebar.checkbox('Enable Daily Returns')
        save_res = st.sidebar.checkbox('Save results')
        plot_save = st.sidebar.checkbox('Save plots')
        st.write('The availability of data')
        st.write('BTC-USD: Historical data is available from 9/14/2014 to present.')
        st.write('ETH-USD: Historical data is available from 11/5/2017 to present.')
        st.write('USDC-USD: Historical data is available from 10/7/2018 to present.')
        st.write('USDT-USD: Historical data is available from 11/5/2017 to present.')
        st.write('DAI-USD: Historical data is available from 11/17/2019 to present.')
        st.write('BNB-USD: Historical data is available from 11/5/2017 to present.')
        st.write('Note: please select data taking into account warmup period. If you will opt for 11/17/2019 there will be'
                 'nan error, because you will not have data to calculate first weights based on previous 50 days .')
        if st.sidebar.button("Run Backtest"):
            model = Full_Pipeline(models,start,end,tickers,warmup,rebalance, predicted_periods, printlog = False, EMA = True)
            model.run_model(save_res, plot_save=False, daily_return = daily_return, plot_show = True)