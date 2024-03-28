from unittest import TestCase
import numpy as np
from ml_pipeline.convolution_2d_layer_nm_model import Convolution2DLayerNMModel, Config

from numpy.testing import (
    assert_allclose,
    assert_almost_equal,
    assert_array_almost_equal,
    assert_array_equal,
    assert_equal,
)

from ml_pipeline.recent_rates import RecentRates


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
            homeostasis_bump_factor=0.1,
            lnr_inhibition_threshold=0.5,
            inhibition_scale_factor=0.0,
            inhibition_reach=1,
            num_out_spikes=3,
        )

        model = Convolution2DLayerNMModel(config)

        assert_array_almost_equal(model._recent_rates.get_rates(), np.full(4, 0.75))

        model._weights[0][0] = 0.6
        model._weights[1][9] = 0.8
        model._weights[2][10] = 0.7
        model._weights[3][24] = 0.7

        result = model.process_frame(np.ones(25))
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

    def test_homeostasis(self):
        config = Config(
            in_size=5,
            conv_kernel_width=3,
            conv_stride=2,
            ltp_step_up=0.0,
            ltp_step_down=0.0,
            recent_rates_half_life=100,
            homeostasis_bump_factor=60,
            lnr_inhibition_threshold=0.5,
            inhibition_scale_factor=0.0,
            inhibition_reach=1,
            num_out_spikes=3,
        )

        model = Convolution2DLayerNMModel(config)

        model._weights = np.zeros((4, 25))
        model._weights[0, 2] = 0.6
        model._weights[1, 3] = 0.56
        model._weights[2, 15] = 0.55

        result = model.process_frame(np.ones(25))
        self.assertEqual(set(result.out_frame), set([0, 1, 2]))

        recent_rates = model._recent_rates.get_rates()

        test_accounting_recent_rates = RecentRates(
            4, config.recent_rates_half_life, 0.75
        )
        test_accounting_recent_rates.update_multi(np.array([0, 1, 2]))

        expected_recent_rates = test_accounting_recent_rates.get_rates()
        assert_almost_equal(recent_rates, expected_recent_rates)
        expected_bumps_1st = (0.75 - expected_recent_rates) * 60

        assert_almost_equal(model._homeostasis_offsets, expected_bumps_1st)
        result = model.process_frame(np.ones(25))
        assert_almost_equal(
            result.v, expected_bumps_1st + np.array([0.6, 0.56, 0.55, 0])
        )
        self.assertEqual(set(result.out_frame), set([0, 1, 2]))

        test_accounting_recent_rates.update_multi(np.array([0, 1, 2]))
        expected_recent_rates = test_accounting_recent_rates.get_rates()
        recent_rates = model._recent_rates.get_rates()
        assert_almost_equal(recent_rates, expected_recent_rates)
        expected_bumps_2nd = (0.75 - expected_recent_rates) * 60

        expected_homeostasis_offsets = expected_bumps_1st + expected_bumps_2nd
        assert_almost_equal(model._homeostasis_offsets, expected_homeostasis_offsets)
        result = model.process_frame(np.ones(25))
        assert_almost_equal(
            result.v,
            expected_homeostasis_offsets + np.array([0.6, 0.56, 0.55, 0]),
        )
        self.assertEqual(set(result.out_frame), set([0, 1, 3]))

    def test_lateral_inhibition_scenario_1(self):
        config = Config(
            in_size=5,
            conv_kernel_width=2,
            conv_stride=1,
            ltp_step_up=0.0,
            ltp_step_down=0.0,
            recent_rates_half_life=2000,
            homeostasis_bump_factor=0.0,
            lnr_inhibition_threshold=0.2,
            inhibition_scale_factor=0.1,
            inhibition_reach=1,
            num_out_spikes=1,
        )

        model = Convolution2DLayerNMModel(config)
        model._weights = np.ones((16, 25))
        model._homeostasis_offsets = np.arange(16) * 0.01

        result = model.map_frame(np.ones(25) * 0.5)

        expected_v = np.array(
            [
                [2 - 0.08, 2.01 - 0.06, 2.02 - 0.06, 2.03 - 0.1 * (2 / 3 - 0.2)],
                [2.04 - 0.04, 2.05 - 0.03, 2.06 - 0.03, 2.07 - 0.02],
                [2.08 - 0.04, 2.09 - 0.03, 2.1 - 0.03, 2.11 - 0.02],
                [2.12 - 0.1 * (1 / 3 - 0.2), 2.13 - 0, 2.14 - 0, 2.15 - 0],
            ]
        ).reshape(16)

        assert_array_almost_equal(result.v, expected_v)

    def test_lateral_inhibition_scenario_2(self):
        config = Config(
            in_size=5,
            conv_kernel_width=2,
            conv_stride=1,
            ltp_step_up=0.0,
            ltp_step_down=0.0,
            recent_rates_half_life=2000,
            homeostasis_bump_factor=0.0,
            lnr_inhibition_threshold=0.0,
            inhibition_scale_factor=0.5,
            inhibition_reach=1,
            num_out_spikes=2,
        )

        model = Convolution2DLayerNMModel(config)
        model._weights = np.zeros_like(model._weights)
        model._homeostasis_offsets = np.array(
            [[0, 0, 0.1, 0], [0.87, 1, 0.9, 0], [0.2, 0.6, 0.3, 0], [0.85, 0, 0, 0]]
        ).reshape(16)

        expected_v = np.array(
            [
                [-1 / 3, -0.4, 0.1 - 0.2, -1 / 3],
                [0.87 - 0.1, 1, 0.9 - 1 / 16, -0.3],
                [0.2 - 0.4, 0.6 - 0.25, 0.3 - 3 / 16, -1 / 5],
                [0.85, -0.4, -0.2, -1 / 6],
            ]
        ).reshape(16)

        result = model.map_frame(np.zeros(25))

        assert_array_almost_equal(result.v, expected_v)
        self.assertEqual(set(result.out_frame), set([5, 12]))

    def test_plasticity(self):
        config = Config(
            in_size=4,
            conv_kernel_width=2,
            conv_stride=1,
            ltp_step_up=0.3,
            ltp_step_down=0.36,
            recent_rates_half_life=2000,
            homeostasis_bump_factor=0.0,
            lnr_inhibition_threshold=0.0,
            inhibition_scale_factor=0.0,
            inhibition_reach=1,
            num_out_spikes=5,
        )

        model = Convolution2DLayerNMModel(config)
        model._weights = np.full_like(model._weights, 0.5)
        model._weights[8, 6] = 0.85
        model._weights[2, 6] = 0.1

        model._homeostasis_offsets = np.array(
            [[0.6, 0.8, 0], [0, 0, 0.9], [0, 0.6, 0.7]]
        ).reshape(9)

        in_frame = np.zeros(16)
        in_frame[[6, 8, 9]] = 0.1
        result = model.process_frame(
            in_frame, tf_spike=np.array([0, 3]), tf_no_spike=np.array([1])
        )

        self.assertEqual(set(result.out_frame), set([0, 1, 5, 7, 8]))

        expected_updated_weights = np.full_like(model._weights, 0.5)
        expected_updated_weights[:, 6] = [0.8, 0.14, 0, 0.8, 0.14, 0.8, 0.14, 0.8, 1]
        expected_updated_weights[:, 8] = [
            0.8,
            0.14,
            0.14,
            0.8,
            0.14,
            0.8,
            0.14,
            0.8,
            0.8,
        ]
        expected_updated_weights[:, 9] = [
            0.8,
            0.14,
            0.14,
            0.8,
            0.14,
            0.8,
            0.14,
            0.8,
            0.8,
        ]

        assert_array_almost_equal(model._weights, expected_updated_weights)
