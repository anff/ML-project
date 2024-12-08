import pandas as pd
import plotly.graph_objects as go
from matplotlib import pyplot as plt
import ta
import numpy as np

from opt import GlobalConst as glb


def read_file(f2p):
    r = 0.2 # ratio of test data
    df = pd.read_csv(f2p)
    df['Date'] = pd.to_datetime(df['Date'])
    df = df[df['Date'].dt.year >= 2018].copy()
    df.index = pd.to_datetime(df['Date'], format='%Y%m%d')
    df = add_indicator(df)

    n_train = int((1-r)*len(df))
    train_df = df.iloc[:n_train].copy()
    # test_df = df.iloc[n_train:].copy()
    # add the test data from the last window
    test_df = df.iloc[n_train-glb.slide_window:].copy()
    return df, train_df, test_df


def add_indicator(df, sma_window=glb.slide_window):
    col_target = 'Close'
    df['sma'] = ta.trend.sma_indicator(df[col_target], sma_window)
    df['ema'] = ta.trend.ema_indicator(df[col_target], sma_window)
    df[col_target] = df[col_target].shift(-1) # To make all the indicator features aligned with target
    df = df.dropna()
    return df


def print_totdata(df, col1, lb1, test_df, col2, lb2):
    plt.figure(figsize=(10, 6))
    plt.plot(df[col1], label=lb1)
    plt.plot(test_df[col2], label=lb2)
    plt.title('Close price')
    plt.xlabel('time', fontsize=15, verticalalignment='top')
    plt.ylabel('close', fontsize=15, horizontalalignment='center')
    plt.legend()
    plt.show()


def genArr_lstm(df, scaler, col_list):
    df = df.iloc[:][col_list]
    df_sc = scaler.fit_transform(df)
    X, y = data_split(df_sc)
    return X, y


def genArr_forlstm(yinput, scaler):
    df_sc = scaler.fit_transform(yinput)
    X, y = data_split(df_sc)
    return X, y


def data_split(sequence, n_timestamp=glb.slide_window):
    X = []
    y = []
    for i in range(len(sequence)):
        end_ix = i + n_timestamp
        if end_ix > len(sequence) - 1:
            break

        if isinstance(sequence, np.ndarray):
            seq_x, seq_y = sequence[i:end_ix], sequence[end_ix]
        else:
            seq_x, seq_y = sequence.iloc[i:end_ix], sequence.iloc[end_ix]
        X.append(seq_x)
        y.append(seq_y)
    return np.array(X), np.array(y)


# def present_oridf(df):
#     fig = make_subplots(rows=1, cols=1)
#     fig.add_trace(go.Ohlc(x=df.Date,
#                           open=df.Open,
#                           high=df.High,
#                           low=df.Low,
#                           close=df.Close,
#                           name='Price'), row=1, col=1)
#     fig.show()


