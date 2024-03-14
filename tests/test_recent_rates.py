from unittest import TestCase
from ml_pipeline.recent_rates import RecentRates


class TestRecentRates(TestCase):
    def test_decay_factor(self):
        expected_decay_factor = 0.99930709299
        sut = RecentRates(10, 1000)
        self.assertAlmostEqual(sut._decay_factor, expected_decay_factor)

        sut._rates[3] = 1.0
        for _ in range(1000):
            sut.update(4)

        rates = sut.get_rates()

        self.assertAlmostEqual(rates[2], 0.0)
        self.assertAlmostEqual(rates[3], 0.5)
        self.assertAlmostEqual(rates[4], 0.5)
