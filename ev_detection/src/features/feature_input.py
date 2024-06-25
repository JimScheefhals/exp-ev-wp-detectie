from dataclasses import dataclass
from functools import cached_property

import pandas as pd
import numpy as np

from ev_detection.src.features.utils.changepoint_detection import changepoint_detection
from ev_detection.src.features.utils.peak_detection import peak_detection_multiple_samples
from ev_detection.src.features.utils.plateau_detection import PlateauDetection


@dataclass
class FeatureInput:
    """
    This class is used to store the input data for the construction of features.
    """
    all_profiles: dict[int, pd.Series]
    datetime: pd.Series
    meta_data: pd.DataFrame

    @cached_property
    def peak_properties(self) -> dict[int, [dict[str, np.ndarray]]]:
        """
        Return the peak properties (prominence and width) for each sample in self.all_profiles.
        """
        return peak_detection_multiple_samples(self.all_profiles)

    @cached_property
    def changepoints(self) -> dict[int, list[int]]:
        """
        Return the changepoints for each sample in self.all_profiles
        """
        return changepoint_detection(self.all_profiles)

    @cached_property
    def plateaus(self) -> dict[int, list[tuple[int, int]]]:
        """
        Calculate the plateaus, separated by the detected changepoints.
        Return the left and right boundaries of the locally maximum plateaus.
        """
        return {
            i: PlateauDetection(sample, self.changepoints[i]).get_bounds_high_plateaus()
            for i, sample in self.all_profiles.items()
        }