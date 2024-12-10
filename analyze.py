import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

import process_data
import build_model
from sklearn import metrics
from opt import GlobalConst as glb


def run_ml(total_df, train_df, test_df):
    cols = ['sma', 'ema']
    X_train = gen_arr(train_df, cols)
    y_train = gen_arr(train_df, ['Close'])

    X_test = gen_arr(test_df, cols)
    model = build_model.build_model()
    model.fit(X_train, y_train)

    y_train_pred = model.predict(X_train)
    train_df['pred_close'] = y_train_pred
    y_pred = model.predict(X_test)
    y_test = gen_arr(test_df, ['Close'])
    test_df['pred_close'] = y_pred
    evaluation_metric(y_test, y_pred)
    process_data.print_totdata(test_df, 'Close', 'golden', test_df, 'pred_close', 'predict')


def run_lstm(total_df, train_df, test_df):
    sc = MinMaxScaler(feature_range=(0, 1))
    col_list = ['Open', 'High', 'Low', 'Close'] #, 'Volume']
    X_train, y_train = process_data.genArr_lstm(train_df, sc, col_list)
    X_test, y_test = process_data.genArr_lstm(test_df, sc, col_list)
    model_par = {'input_size': X_train.shape[-1], 'hidden_size': 50, 'output_size': X_train.shape[-1]}
    model = build_model.build_model(model_par)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test, sc)
    y_test = gen_arr(test_df.iloc[glb.slide_window:], ['Close'])

    evaluation_metric(y_test, y_pred)
    test_df = test_df.iloc[glb.slide_window:]
    test_df['pred_close'] = y_pred
    process_data.print_totdata(test_df, 'Close', 'golden', test_df, 'pred_close', 'predict')


def run_arima_lstm(total_df, train_df, test_df):
    train_y = np.array(train_df['Close']).ravel()
    test_y = np.array(test_df.iloc[glb.slide_window:]['Close']).ravel()
    model_par = {'input_size': 1, 'hidden_size': 50, 'output_size': 1}
    model = build_model.build_model(model_par)
    # Use arima to calculate the first order estimation and residule
    y_arima, y_res = model.proc_arima(train_y, test_y)

    # Next use lstm to predict the residules
    sc = MinMaxScaler(feature_range=(0, 1))
    input_res = np.array(y_res[:len(train_df)]).reshape(-1, 1)
    X_train, y_train = process_data.genArr_forlstm(input_res, sc)
    input_res1 = np.array(y_res[len(train_df)-glb.slide_window:]).reshape(-1, 1)

    X_test, y_test = process_data.genArr_forlstm(input_res1, sc)
    model.fit(X_train, y_train)
    y_pred_res = model.predict(X_test, sc)
    y_pred = []
    for mean, res in zip(y_arima, y_pred_res):
        v = mean + res
        y_pred.append(v)

    evaluation_metric(test_y, y_pred)
    test_df = test_df.iloc[glb.slide_window:]
    test_df['pred_close'] = y_pred
    adf = {'Date': test_df['Date'],
           'Close': test_y}
    adf = pd.DataFrame(adf)
    process_data.print_totdata(adf, 'Close', 'golden', test_df, 'pred_close', 'predict')


def gen_arr(df, cols):
    X = np.array(df[cols])
    return X


def evaluation_metric(y_true, y_hat):
    y_true = np.array(y_true).ravel()
    y_hat = np.array(y_hat).ravel()
    MSE = metrics.mean_squared_error(y_true, y_hat)
    RMSE = MSE**0.5
    MAE = metrics.mean_absolute_error(y_true, y_hat)
    R2 = metrics.r2_score(y_true, y_hat)
    print('MSE: %.5f' % MSE)
    print('RMSE: %.5f' % RMSE)
    print('MAE: %.5f' % MAE)
    print('R2: %.5f' % R2)

