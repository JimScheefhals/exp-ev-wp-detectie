from abc import ABC, abstractmethod

import pandas as pd

from ev_detection.src.features.feature_builder import FeatureBuilder
from ev_detection.src.features.feature_input import FeatureInput
from ev_detection.src.types.feature_names import FeatureName


class ModelFeature(ABC):

    def __init__(self, input: FeatureInput):
        self.input = input
        self.n_profiles = len(self.input.all_profiles) # number of profiles
        self.len_series = self.input.all_profiles[0].shape[0] # number of timesteps in the profiles

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

