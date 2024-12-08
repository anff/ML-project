import ta
import os
import matplotlib.pyplot as plt
from xgboost import plot_importance, plot_tree
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split, GridSearchCV

import opt
import analyze
from opt import GlobalConst as glb
import process_data


def main(f2p, model):
    opt.set_env()
    total_df, train_df, test_df = process_data.read_file(f2p)
    # process_data.print_totdata(total_df, 'Close', 'Training', test_df, 'Close', "Testing")

    if model == 'lstm':
        glb.model = model
        analyze.run_lstm(total_df, train_df, test_df)
    else:
        glb.model = model
        analyze.run_ml(total_df, train_df, test_df)


if __name__ == '__main__':
    f2p = 'data/LKNCY.csv'
    # f2p = 'data/MCD.csv'

    # model: enet, xgb, lstm
    main(f2p, model='lstm')



