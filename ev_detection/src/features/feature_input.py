from dataclasses import dataclass
from functools import cached_property
from scipy.signal import find_peaks

import pandas as pd
import numpy as np

# Parameters for peak_finding
MIN_PROMINENCE = 2 # [kW]
MIN_WIDTH = 1      # [timesteps]
MIN_HEIGHT = 3.7   # [kW]

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
        Return the peak prominences of the profiles.
        """
        print("calculating peak properties")
        properties_all = {}
        for id, sample in self.all_profiles.items():
            peaks_sample, properties_sample = find_peaks(
                sample,
                prominence=MIN_PROMINENCE,
                width=MIN_WIDTH,
                height=MIN_HEIGHT,
            )
            properties_sample["idx"] = peaks_sample
            properties_all[id] = properties_sample
        return properties_all