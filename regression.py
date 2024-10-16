import numpy as np

import analysis
import build_model
from sklearn import metrics

def run():
    X_train, y_train, X_test, y_test = analysis.gen_data()
    model = build_model.build_model()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print(f'y_true = {np.array(y_test)[:5]}')
    print(f'y_pred = {y_pred[:5]}')

    # analysis.show_predict(df)
    rmse = np.sqrt(metrics.mean_squared_error(y_test, y_pred))
    print(f"RMSE: {rmse:.2f}")
    mape = metrics.mean_absolute_percentage_error(y_test, y_pred) * 100
    print(f"MAPE: {mape:.2f}")