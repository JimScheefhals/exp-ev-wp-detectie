import pandas as pd
import numpy as np
from scipy.signal import argrelextrema

class PlateauDetection:
    def __init__(self, signal: pd.Series, changepoints: list[int]):
        self.signal = signal
        self.changepoints = changepoints
        self._find_separators()

    def get_bounds_high_plateaus(self) -> list[tuple[int, int]]:
        """
        Return the left and right indices of the plateaus, separated by the given changepoints, which are higher than
        their direct neighboors.
        """
        return [
            (self.separators[i], self.separators[i+1])
            for i in self._get_local_maxima_plateaus()
        ]

    def _find_separators(self):
        self.separators = [0] + self.changepoints + [len(self.signal)]

    def _get_plateau_average(self):
        return [
            self.signal[left_idx:right_idx].mean()
            for left_idx, right_idx in zip(self.separators[:-1], self.separators[1:])
        ]

    def _get_local_maxima_plateaus(self):
        return argrelextrema(np.array(self._get_plateau_average()), np.greater)[0]


