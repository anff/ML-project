import ta
import os
import matplotlib.pyplot as plt
from xgboost import plot_importance, plot_tree
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler

import opt
import analyze
from opt import GlobalConst as glb
import process_data


def main(f2p, model):
    opt.set_env()
    total_df, train_df, test_df = process_data.read_file(f2p)
    process_data.print_totdata(total_df, 'Close', 'Training', test_df, 'Close', "Testing")

    glb.model = model
    if model == 'lstm':
        analyze.run_lstm(total_df, train_df, test_df)
    elif model == 'arima_lstm':
        analyze.run_arima_lstm(total_df, train_df, test_df)
    else:
        analyze.run_ml(total_df, train_df, test_df)


if __name__ == '__main__':
    f2p = 'data/MCD.csv'
    print(f2p)
    # model: enet, xgb, lstm, arima_lstm
    main(f2p, model='enet')
    main(f2p, model='xgb')
    main(f2p, model='lstm')
    main(f2p, model='arima_lstm')

    f2p = 'data/LKNCY.csv'
    print(f2p)
    # model: enet, xgb, lstm, arima_lstm
    main(f2p, model='enet')
    main(f2p, model='xgb')
    main(f2p, model='lstm')
    main(f2p, model='arima_lstm')




