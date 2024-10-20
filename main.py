import ta
import os
import matplotlib.pyplot as plt
from xgboost import plot_importance, plot_tree
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split, GridSearchCV

import opt
import analyze

from plotly.subplots import make_subplots
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
from opt import GlobalConst as glb


def main():
    f2p = 'data/LKNCY.csv'
    # f2p = 'data/MCD.csv'
    glb.model = 'xgb'
    opt.set_env()
    # analyze.run_flow(f2p)
    analyze.run_flow_ideal(f2p)


if __name__ == '__main__':
    main()



