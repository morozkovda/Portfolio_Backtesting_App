import plotly.graph_objects as go
import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px
import streamlit as st


def plotReturns(df,warmup=0,rebalance_day=0,save=False, show=False):
    fig = px.line(df,title="Cumulative_Returns_warmup_%d_rebal_%d" %(int(warmup), rebalance_day))
    fig.update_layout(coloraxis=dict(colorscale='Viridis'))
    if save == True:
        fig.write_image('graphs/cumreturn_warmup_%d_rebal_%d.png' % (int(warmup), int(rebalance_day)), engine="kaleido",width=1700,height=800)
    if show == True:
        st.plotly_chart(fig)
        # fig.show()


def plotStatsValues(data, title,warmup=0, rebalance_day=0, save=False, show=False):
    labels={"index": "Model", "value": "%s" %title }
    colors = ['lightslategray'] * data.shape[0]
    colors[data.argmax()] = 'green'
    colors[data.argmin()] = 'crimson'
    fig = px.bar(data, barmode='group', title="%s_warmup_%d_rebal_%d" %(title, warmup, rebalance_day), labels=labels)
    fig.update_traces(text=data, texttemplate='%{text:.4f}', textposition='outside', marker_color=colors)
    fig.update_layout(showlegend=False)
    if save == True:
        fig.write_image('graphs/stats_%s_warmup_%d_rebal_%d.png' % (title, int(warmup), rebalance_day),
                        engine = "kaleido", width=1700, height=800)
    if show == True:
        # fig.show()
        st.plotly_chart(fig)


