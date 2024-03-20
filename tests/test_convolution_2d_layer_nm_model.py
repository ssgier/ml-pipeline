from unittest import TestCase
import numpy as np
from ml_pipeline.convolution_2d_layer_nm_model import Convolution2DLayerNMModel, Config

from numpy.testing import (
    assert_allclose,
    assert_almost_equal,
    assert_array_almost_equal,
    assert_array_equal,
)


class TestConvolution2DLayerNMModel(TestCase):
    def test_empty(self):
        config = Config(
            in_size=5,
            conv_kernel_width=3,
            conv_stride=2,
            ltp_step_up=0.3,
            ltp_step_down=0.34,
            recent_rates_half_life=2000,
            homeostasis_bump_factor=0,
            lnr_inhibition_threshold=0.5,
            inhibition_scale_factor=0,
            inhibition_reach=1,
            num_out_spikes=3,
        )

        model = Convolution2DLayerNMModel(config)
        result = model.process_frame(np.zeros(25))
        self.assertEqual(len(result.out_frame), 3)
        assert_array_almost_equal(model._weights, np.zeros((4, 25)))
        assert_almost_equal(result.v, np.zeros(4))

    def test_simple_scenario(self):
        config = Config(
            in_size=5,
            conv_kernel_width=3,
            conv_stride=2,
            ltp_step_up=0.3,
            ltp_step_down=0.34,
            recent_rates_half_life=2000,
            homeostasis_bump_factor=0,
            lnr_inhibition_threshold=0.5,
            inhibition_scale_factor=0.0,
            inhibition_reach=1,
            num_out_spikes=3,
        )

        model = Convolution2DLayerNMModel(config)

        model._weights[0][0] = 0.6
        model._weights[1][9] = 0.8
        model._weights[2][10] = 0.7
        model._weights[3][24] = 0.7

        result = model.map_frame(np.ones(25))
        self.assertEqual(set(result.out_frame), set([1, 2, 3]))
        expected_v = np.array([0.6, 0.8, 0.7, 0.7])
        assert_array_almost_equal(result.v, expected_v)

    def test_basic_convolution(self):
        config = Config(
            in_size=5,
            conv_kernel_width=3,
            conv_stride=2,
            ltp_step_up=0.3,
            ltp_step_down=0.34,
            recent_rates_half_life=2000,
            homeostasis_bump_factor=0,
            lnr_inhibition_threshold=0.5,
            inhibition_scale_factor=0.0,
            inhibition_reach=1,
            num_out_spikes=2,
        )

        model = Convolution2DLayerNMModel(config)
        model._weights = np.tile(np.arange(25, dtype=np.float64) + 1, (4, 1))

        in_frame = np.arange(25, dtype=np.float64) + 1

        result = model.map_frame(in_frame)
        self.assertEqual(set(result.out_frame), set([2, 3]))
        expected_v = np.array([597, 885, 2757, 3405])
        assert_array_almost_equal(result.v, expected_v)
