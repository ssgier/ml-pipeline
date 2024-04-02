from dataclasses import dataclass
from math import sqrt
from ml_pipeline.recent_rates import RecentRates
from ml_pipeline.util import (
    make_convolution_weight_mask,
    make_proximity_weight_mask,
    compute_local_normalized_ranks,
)
import numpy as np


@dataclass
class Config:
    in_size: int
    conv_kernel_width: int
    conv_stride: int
    ltp_step_up: float
    ltp_step_down: float
    recent_rates_half_life: float
    homeostasis_bump_factor: float
    lnr_inhibition_threshold: float
    inhibition_scale_factor: float
    inhibition_reach: int
    num_out_spikes: int


@dataclass
class Result:
    out_frame: np.ndarray
    v: np.ndarray


class Convolution2DLayerNMModel:
    def __init__(self, config: Config) -> None:
        self._config = config
        self._conv_weight_mask = make_convolution_weight_mask(
            config.in_size, config.conv_kernel_width, config.conv_stride
        )
        self._weights = np.zeros_like(self._conv_weight_mask)

        N_out = len(self._weights)
        self._N_out = N_out
        out_size = sqrt(N_out)
        assert out_size.is_integer()
        out_size = int(out_size)
        self._target_rate = config.num_out_spikes / N_out
        self._recent_rates = RecentRates(
            N_out, config.recent_rates_half_life, self._target_rate
        )

        self._homeostasis_offsets = np.zeros(N_out)
        self._proximity_weight_mask = make_proximity_weight_mask(
            out_size, config.inhibition_reach
        )
        self._in_arange = np.arange(self._N_out)

    def get_N_out(self):
        return self._N_out

    def map_frame(self, in_frame: np.ndarray) -> Result:
        v = (
            np.matmul(self._weights * self._conv_weight_mask, in_frame).reshape(
                self._N_out
            )
            + self._homeostasis_offsets
        )

        lnr = compute_local_normalized_ranks(v, self._proximity_weight_mask)

        inhibition = self._config.inhibition_scale_factor * np.maximum(
            lnr - self._config.lnr_inhibition_threshold, 0
        )

        v -= inhibition

        out_frame = np.argpartition(-v, self._config.num_out_spikes)[
            : self._config.num_out_spikes
        ]

        return Result(v=v, out_frame=out_frame)

    def process_frame(
        self,
        in_frame: np.ndarray,
        tf_spike: np.ndarray = np.array([], dtype=np.int64),
        tf_no_spike: np.ndarray = np.array([], dtype=np.int64),
    ) -> Result:
        result = self.map_frame(in_frame)
        out_frame = result.out_frame

        effective_spikes = np.setdiff1d(np.union1d(out_frame, tf_spike), tf_no_spike)
        effective_no_spikes = np.setdiff1d(self._in_arange, effective_spikes)

        cols_to_update = in_frame > 0.0

        if cols_to_update.sum() > 0:
            if len(effective_spikes) > 0:
                ix = np.ix_(effective_spikes, cols_to_update)
                self._weights[ix] += self._config.ltp_step_up
                self._weights[ix] = np.minimum(self._weights[ix], 1.0)
            if len(effective_no_spikes) > 0:
                ix = np.ix_(effective_no_spikes, cols_to_update)
                self._weights[ix] -= self._config.ltp_step_down
                self._weights[ix] = np.maximum(self._weights[ix], 0.0)

        self._recent_rates.update_multi(out_frame)
        target_directions = self._target_rate - self._recent_rates.get_rates()
        homeostasis_bumps = target_directions * self._config.homeostasis_bump_factor
        self._homeostasis_offsets += homeostasis_bumps

        return result
