import pandas as pd


from plotly.subplots import make_subplots
import plotly.graph_objects as go
# Time series decomposition
import statsmodels.api as sm
from mpl_toolkits.mplot3d import Axes3D
from pandas.plotting import scatter_matrix



def gen_data():
    train_df, test_df = process_data()

    # They are trying calculate the indicator
    y_train = train_df['Close'].copy()
    X_train = train_df.drop(['Close'], axis = 1)

    y_test  = test_df['Close'].copy()
    X_test  = test_df.drop(['Close'], axis = 1)

    X_test.head()
    return X_train, y_train, X_test, y_test


def process_data():
    # read csv data
    df = pd.read_csv('data/MCD.csv')

    # I want to crop the time frame starting at 2010 to reduct the amount of data to be processed
    df['Date'] = pd.to_datetime(df['Date'])
    df = df[df['Date'].dt.year >= 2010].copy()
    # df = df.copy()
    df.index = range(len(df))

    df.head()
    print(df.head())
    print(len(df))

    # plot the trend
    fig = make_subplots(rows=2, cols=1)
    fig.add_trace(go.Ohlc(x=df.Date,
                          open=df.Open,
                          high=df.High,
                          low=df.Low,
                          close=df.Close,
                          name='Price'), row=1, col=1)

    fig.add_trace(go.Scatter(x=df.Date, y=df.Volume, name='Volume'), row=2, col=1)

    fig.update(layout_xaxis_rangeslider_visible=False)
    fig.show()

    # plot the data close
    df_close = df[['Date', 'Close']].copy()
    df_close = df_close.set_index('Date')
    decomposition = sm.tsa.seasonal_decompose(df_close, model='multiplicative', period=365)
    decomposition.plot()

    # all created indicators
    df['EMA_9'] = df['Close'].ewm(9).mean().shift()
    df['SMA_5'] = df['Close'].rolling(5).mean().shift()
    df['SMA_10'] = df['Close'].rolling(10).mean().shift()
    df['SMA_15'] = df['Close'].rolling(15).mean().shift()
    df['SMA_30'] = df['Close'].rolling(30).mean().shift()

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.Date, y=df.EMA_9, name='EMA 9'))
    fig.add_trace(go.Scatter(x=df.Date, y=df.SMA_5, name='SMA 5'))
    fig.add_trace(go.Scatter(x=df.Date, y=df.SMA_10, name='SMA 10'))
    fig.add_trace(go.Scatter(x=df.Date, y=df.SMA_15, name='SMA 15'))
    fig.add_trace(go.Scatter(x=df.Date, y=df.SMA_30, name='SMA 30'))
    fig.add_trace(go.Scatter(x=df.Date, y=df.Close, name='Close', opacity=0.2))
    fig.show()

    df['RSI'] = relative_strength_idx(df).fillna(0)
    fig = go.Figure(go.Scatter(x=df.Date, y=df.RSI, name='RSI'))
    fig.show()

    draw_1(df)
    train_df, test_df = process_df(df)
    return train_df, test_df


def process_df(df):
    # this should be problematic
    df['Close'] = df['Close'].shift(-1)
    print(f"Before: {df.shape}")
    df = df.dropna()
    df.index = range(len(df))
    print(f"After: {df.shape}")

    test_size = 0.15
    test_split_idx = int(df.shape[0] * (1 - test_size))

    train_df = df.loc[:test_split_idx].copy()
    test_df = df.loc[test_split_idx + 1:].copy()

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=train_df.Date, y=train_df.Close, name='Training'))
    fig.add_trace(go.Scatter(x=test_df.Date, y=test_df.Close, name='Test'))
    fig.show()

    drop_cols = ['Date', 'Volume', 'Open', 'Low', 'High', 'Adj Close']

    train_df = train_df.drop(drop_cols, axis=1)
    test_df = test_df.drop(drop_cols, axis=1)
    train_df.head()

    scatter_matrix(train_df.iloc[:1000, :], alpha=0.8, figsize=(10, 10), diagonal='hist')
    return train_df, test_df

#
# def show_predict(df):
#     predicted_prices = df.loc[test_split_idx + 1:].copy()
#     predicted_prices['Close'] = y_pred
#
#     fig = make_subplots(rows=2, cols=1)
#     fig.add_trace(go.Scatter(x=df.Date, y=df.Close,
#                              name='Truth',
#                              marker_color='LightSkyBlue'), row=1, col=1)
#
#     fig.add_trace(go.Scatter(x=predicted_prices.Date,
#                              y=predicted_prices.Close,
#                              name='Prediction',
#                              marker_color='MediumPurple'), row=1, col=1)
#
#     fig.add_trace(go.Scatter(x=predicted_prices.Date,
#                              y=y_test,
#                              name='Truth',
#                              marker_color='LightSkyBlue',
#                              showlegend=False), row=2, col=1)
#
#     fig.add_trace(go.Scatter(x=predicted_prices.Date,
#                              y=y_pred,
#                              name='Prediction',
#                              marker_color='MediumPurple',
#                              showlegend=False), row=2, col=1)
#
#     fig.show()


def relative_strength_idx(df, n=14):
    close = df['Close']
    delta = close.diff()
    delta = delta[1:]
    pricesUp = delta.copy()
    pricesDown = delta.copy()
    pricesUp[pricesUp < 0] = 0
    pricesDown[pricesDown > 0] = 0
    rollUp = pricesUp.rolling(n).mean()
    rollDown = pricesDown.abs().rolling(n).mean()
    rs = rollUp / rollDown
    rsi = 100.0 - (100.0 / (1.0 + rs))
    return rsi


def draw_1(df):
    # indicators versus the date
    EMA_12 = pd.Series(df['Close'].ewm(span=12, min_periods=12).mean())
    EMA_26 = pd.Series(df['Close'].ewm(span=26, min_periods=26).mean())
    df['MACD'] = pd.Series(EMA_12 - EMA_26)
    df['MACD_signal'] = pd.Series(df.MACD.ewm(span=9, min_periods=9).mean())

    fig = make_subplots(rows=2, cols=1)
    fig.add_trace(go.Scatter(x=df.Date, y=df.Close, name='Close'), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.Date, y=EMA_12, name='EMA 12'), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.Date, y=EMA_26, name='EMA 26'), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.Date, y=df['MACD'], name='MACD'), row=2, col=1)
    fig.add_trace(go.Scatter(x=df.Date, y=df['MACD_signal'], name='Signal line'), row=2, col=1)
    fig.show()

