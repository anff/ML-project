import numpy as np

import analysis
import analyze
import build_model
from sklearn import metrics

def run_pred(X_train, y_train, X_test):
    model = build_model.build_model()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # rmse = np.sqrt(metrics.mean_squared_error(y_train, y_pred))
    # print(f"RMSE: {rmse:.2f}")
    # mape = metrics.mean_absolute_percentage_error(y_train, y_pred) * 100
    # print(f"MAPE: {mape:.2f}")
    return y_pred


    # analysis.show_predict(df)
