import numpy as np


class RecentRates:
    def __init__(self, N: int, half_life: float, initial_rate: float = 0.0) -> None:
        self._decay_factor = 0.5 ** (1 / half_life)
        self._rates = np.zeros(N)

    def update(self, target: int) -> None:
        self.update_multi(np.array([target]))

    def update_multi(self, targets: np.ndarray) -> None:
        self._rates *= self._decay_factor
        self._rates[targets] += 1.0 * (1 - self._decay_factor)

    def get_rates(self) -> np.ndarray:
        return self._rates.copy()
