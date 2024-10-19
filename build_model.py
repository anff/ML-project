from sklearn.linear_model import ElasticNet
import xgboost as xgb
from opt import GlobalConst as glb


def build_model():
    if glb.model == 'xgb':
        return XGBModel().build()
    elif glb.model == 'enet':
        return ENet().build()


class Model:
    def __init__(self):
        pass

    def build(self):
        return None


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
