import pandas as pd
import ta
from plotly.subplots import make_subplots
import plotly.graph_objects as go


def read_file(f2p):
    r = 0.15 # ratio of test data
    cols = ['Date', 'Close'] # The column to use in train
    df = pd.read_csv(f2p)
    df['Date'] = pd.to_datetime(df['Date'])
    df = df[df['Date'].dt.year >= 2010].copy()
    # present_oridf(df)
    n_train = int((1-r)*len(df))
    train_df = df.iloc[:n_train][cols].copy()
    test_df = df.iloc[n_train+1:][cols].copy()
    return train_df, test_df


def present_oridf(df):
    fig = make_subplots(rows=2, cols=1)
    fig.add_trace(go.Ohlc(x=df.Date,
                          open=df.Open,
                          high=df.High,
                          low=df.Low,
                          close=df.Close,
                          name='Price'), row=1, col=1)
    fig.show()
