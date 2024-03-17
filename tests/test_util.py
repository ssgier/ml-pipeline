from unittest import TestCase
from ml_pipeline.util import (
    make_convolution_weight_mask,
    get_competition_matrix,
    make_proximity_weight_mask,
    compute_lateral_inhibition_addon,
)
from numpy.testing import assert_array_almost_equal
import numpy as np


class TestUtil(TestCase):
    def test_make_convolution_weight_mask(self):
        expected_mask = np.array(
            [
                [
                    [1, 1, 1, 0, 0],
                    [1, 1, 1, 0, 0],
                    [1, 1, 1, 0, 0],
                    [0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0],
                ],
                [
                    [0, 0, 1, 1, 1],
                    [0, 0, 1, 1, 1],
                    [0, 0, 1, 1, 1],
                    [0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0],
                ],
                [
                    [0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0],
                    [1, 1, 1, 0, 0],
                    [1, 1, 1, 0, 0],
                    [1, 1, 1, 0, 0],
                ],
                [
                    [0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0],
                    [0, 0, 1, 1, 1],
                    [0, 0, 1, 1, 1],
                    [0, 0, 1, 1, 1],
                ],
            ]
        ).reshape(4, 25)

        out_mask = make_convolution_weight_mask(5, 3, 2)

        assert_array_almost_equal(out_mask, expected_mask)

    def test_get_competition_matrix(self):
        v = np.array([0.1, 0.6, 0.8, 0.9, 0.1])

        result = get_competition_matrix(v)

        expected = np.array(
            [
                [0, -1, -1, -1, 0],
                [1, 0, -1, -1, 1],
                [1, 1, 0, -1, 1],
                [1, 1, 1, 0, 1],
                [0, -1, -1, -1, 0],
            ]
        )

        assert_array_almost_equal(result, expected)

    def test_make_proximity_weight_mask(self):
        expected_mask = np.array(
            [
                [
                    [1, 1, 0, 0],
                    [1, 1, 0, 0],
                    [0, 0, 0, 0],
                    [0, 0, 0, 0],
                ],
                [
                    [1, 1, 1, 0],
                    [1, 1, 1, 0],
                    [0, 0, 0, 0],
                    [0, 0, 0, 0],
                ],
                [
                    [0, 1, 1, 1],
                    [0, 1, 1, 1],
                    [0, 0, 0, 0],
                    [0, 0, 0, 0],
                ],
                [
                    [0, 0, 1, 1],
                    [0, 0, 1, 1],
                    [0, 0, 0, 0],
                    [0, 0, 0, 0],
                ],
                [
                    [1, 1, 0, 0],
                    [1, 1, 0, 0],
                    [1, 1, 0, 0],
                    [0, 0, 0, 0],
                ],
                [
                    [1, 1, 1, 0],
                    [1, 1, 1, 0],
                    [1, 1, 1, 0],
                    [0, 0, 0, 0],
                ],
                [
                    [0, 1, 1, 1],
                    [0, 1, 1, 1],
                    [0, 1, 1, 1],
                    [0, 0, 0, 0],
                ],
                [
                    [0, 0, 1, 1],
                    [0, 0, 1, 1],
                    [0, 0, 1, 1],
                    [0, 0, 0, 0],
                ],
                [
                    [0, 0, 0, 0],
                    [1, 1, 0, 0],
                    [1, 1, 0, 0],
                    [1, 1, 0, 0],
                ],
                [
                    [0, 0, 0, 0],
                    [1, 1, 1, 0],
                    [1, 1, 1, 0],
                    [1, 1, 1, 0],
                ],
                [
                    [0, 0, 0, 0],
                    [0, 1, 1, 1],
                    [0, 1, 1, 1],
                    [0, 1, 1, 1],
                ],
                [
                    [0, 0, 0, 0],
                    [0, 0, 1, 1],
                    [0, 0, 1, 1],
                    [0, 0, 1, 1],
                ],
                [
                    [0, 0, 0, 0],
                    [0, 0, 0, 0],
                    [1, 1, 0, 0],
                    [1, 1, 0, 0],
                ],
                [
                    [0, 0, 0, 0],
                    [0, 0, 0, 0],
                    [1, 1, 1, 0],
                    [1, 1, 1, 0],
                ],
                [
                    [0, 0, 0, 0],
                    [0, 0, 0, 0],
                    [0, 1, 1, 1],
                    [0, 1, 1, 1],
                ],
                [
                    [0, 0, 0, 0],
                    [0, 0, 0, 0],
                    [0, 0, 1, 1],
                    [0, 0, 1, 1],
                ],
            ]
        ).reshape(16, 16)

        out_mask = make_proximity_weight_mask(4, 1)

        assert_array_almost_equal(out_mask, expected_mask)

    def test_compute_lateral_inhibition_addon(self):
        v = np.array([[0.1, 0.6, 0.2], [0.1, 0.8, 0.0], [0.9, 0.3, 0.9]]).reshape(9)
        expected_addon = np.array([[-2, 3, -1], [-4, 4, -5], [3, -1, 3]]).reshape(9)
        proximity_weight_mask = make_proximity_weight_mask(3, 1)
        addon = compute_lateral_inhibition_addon(v, proximity_weight_mask)
        assert_array_almost_equal(addon, expected_addon)
