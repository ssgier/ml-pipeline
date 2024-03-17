from unittest import TestCase
from ml_pipeline.util import make_convolution_weight_mask, get_competition_matrix
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
