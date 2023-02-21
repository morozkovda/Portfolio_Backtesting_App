import plotly.graph_objects as go
import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px

def plotReturns(df):
    fig = px.line(df)
    fig.show()


df_returns = pd.read_csv('data/model_returns.csv', index_col=0)

plotReturns(df_returns)