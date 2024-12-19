from sklearn.linear_model import ElasticNet
import xgboost as xgb
from opt import GlobalConst as glb
import torch.nn as nn
import torch.optim as optim
import torch
import numpy as np
import copy
import statsmodels.api as sm


def build_model(model_par=None):
    if glb.model == 'xgb':
        return XGBModel().build()
    elif glb.model == 'enet':
        return ENet().build()
    elif glb.model == 'lstm':
        proto = BuildLstm()
        proto.set(model_par)
        return proto.build()
    elif glb.model == 'arima_lstm':
        proto = BuildArima_Lstm()
        proto.set(model_par)
        return proto.build()


class Model:
    def __init__(self):
        pass

    def build(self):
        return None

    def set(self, model_par=None):
        self.model_par = model_par


class ENet(Model):
    def build(self):
        # ElasticNet model
        ENet = ElasticNet(
            alpha=0.1,
            l1_ratio=0.1,
            max_iter=100
        )
        return ENet


class XGBModel(Model):
    def build(self):
        model = xgb.XGBRegressor(max_depth=5, n_estimators=300, learning_rate=0.05)
        return model


class BuildLstm(Model):
    def build(self):
        return self

    def shuffle(self, X, y):
        indices = np.random.permutation(X.shape[0])
        X_shuffled = X[indices]
        y_shuffled = y[indices]
        return X_shuffled, y_shuffled

    def fit(self, X_train, y_train):
        # shuffle samples for later train/vaild split
        X_train, y_train = self.shuffle(X_train, y_train)
        model = LSTMModel(**(self.model_par))
        model.init_weights(model)
        mstate = copy.deepcopy(model.state_dict())

        dt = {
            'learning_rate': [0.01, 0.02, 0.03, 0.05]
        }
        hyper = {}
        for k, vlist in dt.items():
            par = {}
            loss = []
            for v in vlist:
                par[k] = v
                model.load_state_dict(mstate)
                aloss = self.fit_gridcv(X_train, y_train, model, par)
                loss.append(aloss)
            tid = loss.index(min(loss))
            thep = vlist[tid]
            hyper[k] = thep
        model.load_state_dict(mstate)
        self.fit_gridcv(X_train, y_train, model, hyper)
        self.model = model

    def fit_gridcv(self, X_train, y_train, model, par_dict):
        # model.apply(model.init_weights)
        n_train = int(0.7 * len(X_train))
        X_tr = X_train[:n_train]
        y_tr = y_train[:n_train]
        X_val = X_train[n_train:]
        y_val = y_train[n_train:]

        n_epochs = 50
        optimizer = optim.Adam(model.parameters(), lr=par_dict['learning_rate'])
        criterion = nn.MSELoss()  # Mean squared error loss for regression
        train_losses = []
        val_losses = []
        for epoch in range(n_epochs):
            model.train()  # Set the model to training mode
            optimizer.zero_grad()  # Zero the gradients
            outputs = model(torch.Tensor(X_tr))  # Convert to tensor if needed
            loss = criterion(outputs, torch.Tensor(y_tr))  # Compute loss
            loss.backward()
            optimizer.step()

            # Validation loss calculation (on test data)
            model.eval()  # Set the model to evaluation mode
            with torch.no_grad():
                val_output = model(torch.Tensor(X_val))
                val_loss = criterion(val_output, torch.Tensor(y_val))

            # Store losses for later plotting
            train_losses.append(loss.item())
            val_losses.append(val_loss.item())
            # if (epoch + 1) % 10 == 0:
            #     print(f'Epoch {epoch + 1}/{n_epochs}, Train Loss: {loss.item()}, Val Loss: {val_loss.item()}')
        return val_losses[-1]

    def predict(self, X_test, sc=None):
        model = self.model
        model.eval()
        pred = model(torch.Tensor(X_test))
        pred = pred.detach().numpy()
        pred = sc.inverse_transform(pred)
        y_pred = np.array(pred[:, 0]).flatten().tolist()
        return y_pred


class BuildArima_Lstm(BuildLstm):
    def build(self):
        self.model = LSTMModel(**(self.model_par))
        return self

    def proc_arima(self, y_train, y_test):
        history = list(copy.deepcopy(y_train))
        predictions = []
        for t in range(len(y_test)):
            model = sm.tsa.ARIMA(history, order=(2, 1, 0)).fit()
            yhat = model.forecast()
            yhat = np.float(yhat[0])
            predictions.append(yhat)
            obs = y_test[t]
            history.append(obs)

        model_res = sm.tsa.ARIMA(history, order=(2, 1, 0)).fit()
        return predictions, model_res.resid


class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.dense = nn.Linear(hidden_size, output_size)

    def init_weights(self, m):
        if isinstance(m, nn.LSTM):
            for name, param in m.named_parameters():
                if 'weight_ih' in name:
                    # Initialize input-hidden weights with Xavier Uniform
                    nn.init.xavier_uniform_(param.data)
                elif 'weight_hh' in name:
                    # Initialize hidden-hidden weights with Orthogonal initialization
                    nn.init.orthogonal_(param.data)
                elif 'bias' in name:
                    # Initialize biases to zero
                    nn.init.constant_(param.data, 0)

    def forward(self, x):
        # Pass through LSTM layer
        lstm_out, (h_n, c_n) = self.lstm(x)
        # Only take the output from the last time step
        last_out = lstm_out[:, -1, :]
        # Pass through the dense layer
        out = self.dense(last_out)
        return out