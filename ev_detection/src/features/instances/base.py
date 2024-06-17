from abc import ABC, abstractmethod

import pandas as pd

from ev_detection.src.types.feature_names import FeatureName


class ModelFeature(ABC):

    def __init__(self, all_profiles: list[pd.Series]):
        self.all_profiles = all_profiles
        self.n_profiles = len(all_profiles) # number of profiles
        self.len_series = all_profiles[0].shape[0] # number of timesteps in the profiles

    @property
    @abstractmethod
    def type(self) -> FeatureName:
        pass

    @abstractmethod
    def get(self) -> pd.Series:
        """
        Return feature values
        """
        pass

