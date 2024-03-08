from unittest import TestCase
import numpy as np
from ml_pipeline.basic_nm_model import BasicNMModel, Config
from numpy.testing import (
    assert_allclose,
    assert_almost_equal,
    assert_array_almost_equal,
    assert_array_equal,
)


class TestBasicNMModel(TestCase):
    def test_empty(self):
        config = Config(
            N_in=2, N_out=3, ltp_step_up=0.3, ltp_step_down=0.34, N_repeat=1
        )
        model = BasicNMModel(config)
        out = model._process_frame(np.zeros(config.N_in))
        self.assertEqual(out, 0)
        assert_array_almost_equal(model._weights, np.zeros((config.N_out, config.N_in)))

    def test_plasticity(self):
        config = Config(
            N_in=2, N_out=3, ltp_step_up=0.3, ltp_step_down=0.34, N_repeat=1
        )
        in_frame = np.zeros(config.N_in)
        in_frame[0] = 0.6

        tf_spike = np.array([1])
        tf_no_spike = np.array([2])

        config = Config(
            N_in=2, N_out=3, ltp_step_up=0.3, ltp_step_down=0.34, N_repeat=1
        )
        model = BasicNMModel(config)
        model._weights = np.full_like(model._weights, 0.5)

        model._process_frame(in_frame, tf_spike, tf_no_spike)
        expected_weights = np.array([[0.5, 0.5], [0.8, 0.5], [0.16, 0.5]])
        assert_array_almost_equal(model._weights, expected_weights)

        model._process_frame(in_frame, tf_spike, tf_no_spike)
        expected_weights = np.array([[0.5, 0.5], [1.0, 0.5], [0.0, 0.5]])
        assert_array_almost_equal(model._weights, expected_weights)

    def test_fit(self):
        X = np.array([[0.0, 0.6]])
        y = np.array([1])

        config = Config(
            N_in=2, N_out=3, ltp_step_up=0.3, ltp_step_down=0.34, N_repeat=1
        )
        model = BasicNMModel(config)
        model._weights = np.full_like(model._weights, 0.5)
        model.fit(X, y)

        expected_weights = np.array([[0.5, 0.16], [0.5, 0.8], [0.5, 0.16]])
        assert_array_almost_equal(model._weights, expected_weights)

    def test_repeat(self):
        X = np.array([[0.0, 0.6]])
        y = np.array([1])

        config = Config(
            N_in=2, N_out=3, ltp_step_up=0.3, ltp_step_down=0.34, N_repeat=2
        )

        model = BasicNMModel(config)
        model._weights = np.full_like(model._weights, 0.5)
        model.fit(X, y)

        expected_weights = np.array([[0.5, 0.0], [0.5, 1], [0.5, 0.0]])
        assert_array_almost_equal(model._weights, expected_weights)

        y_predict = model.predict(X)
        assert_array_equal(y_predict, np.array([1]))
