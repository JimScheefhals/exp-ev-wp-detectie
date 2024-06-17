import pandas as pd

from ev_detection.src.features.instances.base import ModelFeature
from ev_detection.src.types.feature_names import FeatureName


class MaxFeature(ModelFeature):

    @property
    def type(self) -> FeatureName:
        return FeatureName.MAX

    def get(self) -> pd.Series:
        return pd.Series([
            profile.max()
            for profile in self.all_profiles
        ])
