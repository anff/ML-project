import pandas as pd
import numpy as np
from xgboost import XGBClassifier, XGBRegressor
from sklearn.metrics import accuracy_score
import ta
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

df = pd.read_csv("/content/drive/MyDrive/ISU2024/ML_24_01/ML_Proje/archive/MCD.csv")
scaler = MinMaxScaler()
scaled = scaler.fit_transform(df[["Close"]])
df["Close"] = scaled


# print(df)

df['Date'] = pd.to_datetime(df['Date'])  # 'Date' 열을 datetime으로 변환
df1 = df[df['Date'].dt.year >= 2010].copy()  # 2010년 이후의 데이터를 필터링
df1.reset_index(inplace=True)  # 인덱스를 리셋
df = df1  # 필터링된 데이터를 df로 저장


def Make_Indicator(df):
  global sma, ema, sto_k, sto_d, rsi, macd, disparity

  sma = ta.trend.sma_indicator(df["Close"], 12)

  ema = ta.trend.ema_indicator(df["Close"], 12)

  sto_k = ta.momentum.stochrsi_k(df["Close"])
  sto_d = ta.momentum.stochrsi_d(df["Close"])

  rsi = ta.momentum.rsi(df["Close"],14)

  macd = ta.trend.macd(df["Close"], 13, 26)

  df["MA"] = df["Close"].rolling(20).mean()
  disparity = 100*(df["Close"]/df["MA"])


def Concat_ind(dataFrame):
  global x, y

  x = np.zeros((len(df)-40-1, 8))
  idx = 0

  for i in range(40, len(df)-1):
    x[idx][0] = sma[i]
    x[idx][1] = ema[i]
    x[idx][2] = sto_k[i]
    x[idx][3] = sto_d[i]
    x[idx][4] = rsi[i]
    x[idx][5] = macd[i]
    x[idx][6] = disparity[i]
    x[idx][7] = df["Close"][i]

    y = np.zeros((len(df)-40-1, 8))
    idx2 = 0
    for j in range(40, len(df)-1):
      y[idx2][0] = df["Close"][j+1]
      idx2 += 1

# window
def Slice_Window(size):
  global X, Y
  X = []
  Y = []
  size = size
  index = 0

  for i in range(0, len(x)-size+1):
      x_1 = []
      y_1 = []

      for j in range(size):
          x_1.append(x[i+j][0])
          x_1.append(x[i+j][1])
          x_1.append(x[i+j][2])
          x_1.append(x[i+j][3])
          x_1.append(x[i+j][4])
          x_1.append(x[i+j][5])
          x_1.append(x[i+j][6])
          x_1.append(x[i+j][7])

      y_1.append(y[i+(size-1)][0])
      X.append(x_1)
      Y.append(y_1)

  print(np.shape(X))



Make_Indicator(df)
Concat_ind(df)
Slice_Window(10)


# model, prediction
model = XGBRegressor(max_depth=5, n_estimators=300, learning_rate=0.05)

model.fit(X[:2900], Y[:2900])

y_pred = model.predict(X[2900:])




# accuracy

same_count = 0
Y2 = Y[1899:]

for i in range(len(y_pred)):
    if i > 0:
        cha = y_pred[i] - y_pred[i-1]  # 예측값의 변화량
        cha2 = Y2[i][0] - Y2[i-1][0]  # 실제값의 변화량

        if cha >= 0 and cha2 >= 0:
            same_count += 1
        if cha < 0 and cha2 < 0:
            same_count += 1

acc = round(same_count / (len(y_pred)-1) * 100, 2)
print("Accuracy : {}%".format(acc))
