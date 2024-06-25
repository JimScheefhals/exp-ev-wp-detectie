from dataclasses import dataclass
from functools import cached_property

import pandas as pd
import numpy as np

from ev_detection.src.features.utils.peak_detection import peak_detection_multiple_samples


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