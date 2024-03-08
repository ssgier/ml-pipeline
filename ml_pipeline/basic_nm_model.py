from sklearn.base import BaseEstimator, ClassifierMixin
from dataclasses import dataclass
import numpy as np


@dataclass
class Config:
    N_in: int
    N_out: int
    ltp_step_up: float
    ltp_step_down: float
    N_repeat: int


class BasicNMModel(BaseEstimator, ClassifierMixin):
    def __init__(self, config: Config) -> None:
        self._config = config
        self._weights = np.zeros((config.N_out, config.N_in))
        #self._normalize_weights()
        super().__init__()

    def fit(self, X, y):
        for _ in range(self._config.N_repeat):
            for x, target in zip(X, y):
                self._train_single(x, target)
        return self

    def predict(self, X):
        return np.array([self._predict_single(x) for x in X])

    def score(self, X, y):
        predictions = self.predict(X)
        return np.mean(predictions == y)

    def _train_single(self, in_frame: np.ndarray, target: int):
        tf_spike = np.array([target])
        tf_no_spike = np.concatenate(
            [np.arange(target), np.arange(target + 1, self._config.N_out)]
        )
        self._process_frame(in_frame, tf_spike, tf_no_spike)

    def _predict_single(self, x) -> int:
        return np.argmax(np.matmul(self._weights, x).reshape(self._config.N_out)).item()

    def _normalize_weights(self) -> None:
        rowsums = self._weights.sum(axis=1, keepdims=True)
        self._weights = np.where(
            rowsums != 0, self._weights / rowsums, 1 / self._weights.shape[1]
        )

    def _process_frame(
        self,
        in_frame: np.ndarray,
        tf_spike: np.ndarray = np.array([]),
        tf_no_spike: np.ndarray = np.array([]),
    ) -> int:
        out = self._predict_single(in_frame)

        # ltp
        cols_to_update = in_frame > 0.5
        if len(tf_spike) > 0:
            ix = np.ix_(tf_spike, cols_to_update)
            self._weights[ix] += self._config.ltp_step_up
            self._weights[ix] = np.minimum(self._weights[ix], 1.0)
        if len(tf_no_spike) > 0:
            ix = np.ix_(tf_no_spike, cols_to_update)
            self._weights[ix] -= self._config.ltp_step_down
            self._weights[ix] = np.maximum(self._weights[ix], 0.0)

        #self._normalize_weights()

        return out
