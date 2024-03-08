from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import cross_val_score
import numpy as np


class DummyModel(BaseEstimator, ClassifierMixin):
    def fit(self, X, y):
        return self

    def predict(self, X):
        return np.ones(X.shape[0], dtype=np.int64)

    def score(self, X, y):
        predictions = self.predict(X)
        return np.mean(predictions == y)
