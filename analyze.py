import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import ta
import plotly.graph_objects as go
import shap

import process_data
import regression
import build_model
from sklearn.metrics import r2_score


def run_flow(f2p):
    # use just the past history data
    train_df, test_df = process_data.read_file(f2p)
    train_df, app_df = add_indicator(train_df)

    cols = ['sma', 'ema']
    X_train = gen_arr(train_df, cols)
    y_train = gen_arr(train_df, ['Close'])
    X_test = gen_arr(app_df, cols)
    model = build_model.build_model()
    model.fit(X_train, y_train)
    y_train = list(y_train.ravel())
    y_pred = []
    y_test = gen_arr(test_df, ['Close'])
    for i in range(len(y_test)):
        y = model.predict(X_test)
        y_pred.append(y[0])
        y_train.append(y[0])
        _, df2 = gen_next(y_train)
        X_test = gen_arr(df2, cols)

    fig = go.Figure()
    df = pd.concat([train_df, test_df])
    fig.add_trace(go.Scatter(x=df.Date, y=df.Close, name='golden'))
    fig.add_trace(go.Scatter(x=test_df.Date, y=y_pred, name='pred'))
    fig.show()


def gen_next(y_train, nsam=20):
    # y_train, list
    col_target = 'Close'
    df = pd.DataFrame()
    df[col_target] = y_train[-nsam:]
    df1, df2 = add_indicator(df) # df1, add to train_df, df2: next predict
    return df1, df2


def gen_arr(df, cols):
    X = np.array(df[cols])
    return X


def add_indicator(df):
    col_target = 'Close'
    sma_window = 20
    df['sma'] = ta.trend.sma_indicator(df[col_target], sma_window)
    df['ema'] = ta.trend.ema_indicator(df[col_target], sma_window)
    df[col_target] = df[col_target].shift(-1) # To make all the indicator features aligned with target
    df_app = df.iloc[[-1]]
    df = df.dropna()
    return df, df_app


def make_indicator(df, win=12, winfast=13, winslow=26):
    sma = ta.trend.sma_indicator(df["Close"], win)
    ema = ta.trend.ema_indicator(df["Close"], win)
    sto_k = ta.momentum.stochrsi_k(df["Close"])

    sto_d = ta.momentum.stochrsi_d(df["Close"])

    rsi = ta.momentum.rsi(df["Close"], win+2) # Need check !!!
    macd = ta.trend.macd(df["Close"], winfast, winslow)
    df["MA"] = df["Close"].rolling(20).mean()

    disparity = 100*(df["Close"]/df["MA"])
    indicators = (sma, ema, sto_k, sto_d, rsi, macd, disparity, df['Close'])
    return indicators


def run_flow_ideal(f2p):
    train_df, test_df = process_data.read_file(f2p)
    train_df, _ = add_indicator(train_df)

    cols = ['sma', 'ema']
    X_train = gen_arr(train_df, cols)
    y_train = gen_arr(train_df, ['Close'])

    test_df, _ = add_indicator(test_df)
    X_test = gen_arr(test_df, cols)
    model = build_model.build_model()
    model.fit(X_train, y_train)

    y_train_pred = model.predict(X_train)
    train_df['pred_close'] = y_train_pred
    y_pred = model.predict(X_test)
    y_test = gen_arr(test_df, ['Close'])
    test_df['pred_close'] = y_pred

    fig = go.Figure()
    df = pd.concat([train_df, test_df])
    fig.add_trace(go.Scatter(x=df.Date, y=df.Close, name='golden'))
    # fig.add_trace(go.Scatter(x=train_df.Date, y=train_df.Close, name='train'))
    fig.add_trace(go.Scatter(x=test_df.Date, y=y_pred, name='predict'))
    fig.show()

    # explainer = shap.Explainer(model, X_train)
    # shap_values = explainer(X_test)
    # Visualize the SHAP values for the first prediction
    # shap.initjs()
    # shap.summary_plot(shap_values, X_test)

    acc_rmse = calculate_rmse(y_test, y_pred)
    acc_mape = calculate_mape(y_test, y_pred)
    acc_r2 = r2_score(y_test, y_pred)
    print('Accuracy rmse: %f; mape %f; r2: %f' % (acc_rmse, acc_mape, acc_r2))


#### Calculate the metrics RMSE and MAPE ####
def calculate_rmse(y_true, y_pred):
    """
    Calculate the Root Mean Squared Error (RMSE)
    """
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    return rmse


def calculate_mape(y_true, y_pred):
    """
    Calculate the Mean Absolute Percentage Error (MAPE) %
    """
    y_pred, y_true = np.array(y_pred), np.array(y_true)
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    return mape

#
# def process_data11(f2p):
#     df = pd.read_csv(f2p)
#     df['Date'] = pd.to_datetime(df['Date'])
#     df = df[df['Date'].dt.year >= 2010].copy()
#     print(df)
#     print(df.head())
#     exit()
#     # df1.reset_index(inplace=True)
#     # df = df1
#
#     # 1. scale
#     scaler = MinMaxScaler()
#     scaled = scaler.fit_transform(df[["Close"]])
#     # print(df.iloc[0])
#     df["Close"] = scaled
#
#     # 2. add indicator
#     indicator = make_indicator(df)
#     x, y = concat_ind(indicator)
#
#     train_x, train_y = slice_window(x, y)
#
#
#
# def concat_ind(indicator):
#     sma, ema, sto_k, sto_d, rsi, macd, disparity, close = indicator
#     x = np.zeros((len(sma)-40-1, 8))
#     idx = 0
#     for i in range(40, len(sma)-1):
#         x[idx][0] = sma[i]
#         x[idx][1] = ema[i]
#         x[idx][2] = sto_k[i]
#         x[idx][3] = sto_d[i]
#         x[idx][4] = rsi[i]
#         x[idx][5] = macd[i]
#         x[idx][6] = disparity[i]
#         x[idx][7] = close[i]
#
#     y = np.zeros((len(sma)-40-1, 8))
#     idx2 = 0
#     for j in range(40, len(sma)-1):
#       y[idx2][0] = close[j+1]
#       idx2 += 1
#     return x, y
#
#
# def slice_window(x, y, size=12):
#     X = []
#     Y = []
#     for i in range(0, len(x)-size+1):
#         x_1 = []
#         y_1 = []
#         for j in range(size):
#             x_1.append(x[i+j][0])
#             x_1.append(x[i+j][1])
#             x_1.append(x[i+j][2])
#             x_1.append(x[i+j][3])
#             x_1.append(x[i+j][4])
#             x_1.append(x[i+j][5])
#             x_1.append(x[i+j][6])
#             x_1.append(x[i+j][7])
#             print(x_1)
#         print(len(x_1), x_1)
#         exit()
#         y_1.append(y[i+(size-1)][0])
#         X.append(x_1)
#         Y.append(y_1)
#
#     print(np.shape(X))
#     return X, Y


#
# Make_Indicator(df)
# Concat_ind(df)
# Slice_Window(10)

#
# # accuracy
#
# same_count = 0
# Y2 = Y[1899:]
#
# for i in range(len(y_pred)):
#     if i > 0:
#         cha = y_pred[i] - y_pred[i-1]  # 예측값의 변화량
#         cha2 = Y2[i][0] - Y2[i-1][0]  # 실제값의 변화량
#
#         if cha >= 0 and cha2 >= 0:
#             same_count += 1
#         if cha < 0 and cha2 < 0:
#             same_count += 1
#
# acc = round(same_count / (len(y_pred)-1) * 100, 2)
# print("Accuracy : {}%".format(acc))
