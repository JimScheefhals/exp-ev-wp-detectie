from dataclasses import dataclass
from functools import cached_property
from scipy.signal import find_peaks

import pandas as pd
import numpy as np

# Parameters for peak_finding
MIN_PROMINENCE = 2 # [kW]
MIN_WIDTH = 2      # [timesteps]
MIN_HEIGHT = 3.7   # [kW]

@dataclass
class FeatureInput:
    """
    This class is used to store the input data for the construction of features.
    """
    all_profiles: list[pd.Series]

    @cached_property
    def peak_properties(self) -> list[dict[str, np.ndarray]]:
        """
        Return the peak prominences of the profiles.
        """
        print("calculating peak properties")
        properties_all = []
        for sample in self.all_profiles:
            _, properties_sample = find_peaks(
                sample,
                prominence=MIN_PROMINENCE,
                width=MIN_WIDTH,
                height=MIN_HEIGHT,
            )
            properties_all.append(properties_sample)
        return properties_all