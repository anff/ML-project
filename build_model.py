from sklearn.linear_model import ElasticNet


def build_model():
    # ElasticNet model
    ENet = ElasticNet(
        alpha=0.1,
        l1_ratio=0.,
        max_iter=1000
    )
    return ENet



