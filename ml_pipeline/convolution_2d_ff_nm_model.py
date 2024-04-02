from sklearn.base import BaseEstimator, ClassifierMixin
from dataclasses import dataclass
from ml_pipeline.recent_rates import RecentRates
from ml_pipeline.convolution_2d_layer_nm_model import Convolution2DLayerNMModel
from typing import List
import numpy as np


@dataclass
class Config:
    N_in: int
    N_out: int
    N_repeat: int


class Convolution2DFFNMModel(BaseEstimator, ClassifierMixin):
    def __init__(self, config: Config, layers: List[Convolution2DLayerNMModel]) -> None:
        self._config = config
        self._layers = layers
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

    def _train_single(self, x: np.ndarray, target: int):
        in_frame = x
        out_frame = np.array([])

        for layer in self._layers[:-1]:
            out_frame = layer.process_frame(in_frame).out_frame
            in_frame = np.zeros(layer.get_N_out())
            in_frame[out_frame] = 1

        tf_spike = np.array([target])
        tf_no_spike = np.concatenate(
            [np.arange(target), np.arange(target + 1, self._config.N_out)]
        )

        self._layers[-1].process_frame(
            in_frame, tf_spike=tf_spike, tf_no_spike=tf_no_spike
        )

    def _predict_single(self, x) -> int:
        in_frame = x
        out_frame = np.array([])

        for layer in self._layers:
            out_frame = layer.map_frame(in_frame).out_frame
            in_frame = np.zeros(layer.get_N_out())
            in_frame[out_frame] = 1

        assert len(out_frame) == 1

        return out_frame[0].item()
