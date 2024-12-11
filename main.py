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
    tt = os.path.splitext(os.path.basename(f2p))[0]
    tt = model + '-' + tt
    print('\n', tt)
    opt.set_env()
    total_df, train_df, test_df = process_data.read_file(f2p)
    # process_data.print_totdata(total_df, 'Close', 'Training', test_df, 'Close', "Testing")

    glb.model = model
    if model == 'lstm':
        test_df1 = analyze.run_lstm(total_df, train_df, test_df)
    elif model == 'arima_lstm':
        test_df1 = analyze.run_arima_lstm(total_df, train_df, test_df)
    else:
        test_df1 = analyze.run_ml(total_df, train_df, test_df)

    process_data.print_totdata(test_df1, 'Close', 'golden', test_df1, 'pred_close', 'predict', tt, show=False)


if __name__ == '__main__':
    f2p = 'data/BRK-A.csv'
    print(f2p)
    # model: enet, xgb, lstm, arima_lstm
    main(f2p, model='enet')
    main(f2p, model='xgb')
    main(f2p, model='lstm')
    main(f2p, model='arima_lstm')

    f2p = 'data/DPZ.csv'
    print(f2p)
    # model: enet, xgb, lstm, arima_lstm
    main(f2p, model='enet')
    main(f2p, model='xgb')
    main(f2p, model='lstm')
    main(f2p, model='arima_lstm')

    f2p = 'data/MCD.csv'
    print(f2p)
    main(f2p, model='enet')
    main(f2p, model='xgb')
    main(f2p, model='lstm')
    main(f2p, model='arima_lstm')

    f2p = 'data/QSR.csv'
    print('\n', f2p)
    main(f2p, model='enet')
    main(f2p, model='xgb')
    main(f2p, model='lstm')
    main(f2p, model='arima_lstm')

    f2p = 'data/WEN.csv'
    print(f2p)
    main(f2p, model='enet')
    main(f2p, model='xgb')
    main(f2p, model='lstm')
    main(f2p, model='arima_lstm')

    f2p = 'data/DNUT.csv'
    print('\n', f2p)
    main(f2p, model='enet')
    main(f2p, model='xgb')
    main(f2p, model='lstm')
    main(f2p, model='arima_lstm')

    f2p = 'data/LKNCY.csv'
    print(f2p)
    main(f2p, model='enet')
    main(f2p, model='xgb')
    main(f2p, model='lstm')
    main(f2p, model='arima_lstm')

    f2p = 'data/PZZA.csv'
    print('\n', f2p)
    main(f2p, model='enet')
    main(f2p, model='xgb')
    main(f2p, model='lstm')
    main(f2p, model='arima_lstm')

    f2p = 'data/SBUX.csv'
    print(f2p)
    main(f2p, model='enet')
    main(f2p, model='xgb')
    main(f2p, model='lstm')
    main(f2p, model='arima_lstm')

    f2p = 'data/YUM.csv'
    print('\n', f2p)
    main(f2p, model='enet')
    main(f2p, model='xgb')
    main(f2p, model='lstm')
    main(f2p, model='arima_lstm')
