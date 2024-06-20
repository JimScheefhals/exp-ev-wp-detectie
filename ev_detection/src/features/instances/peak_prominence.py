import pandas as pd
import numpy as np

from ev_detection.src.features.instances.base import ModelFeature
from ev_detection.src.types.feature_names import FeatureName


class MaxPeakProminenceFeature(ModelFeature):

    @property
    def type(self) -> FeatureName:
        return FeatureName.MAX_PEAK_PROMINENCE

    def get(self) -> pd.Series:
        return pd.Series([
            prop["prominences"].max(initial=0)
            for prop in self.input.peak_properties
        ])


class MeanPeakProminenceFeature(ModelFeature):

    @property
    def type(self) -> FeatureName:
        return FeatureName.MEAN_PEAK_PROMINENCE

    def get(self) -> pd.Series:
        return pd.Series([
            np.append(prop["prominences"], 0).mean()
            for prop in self.input.peak_properties
        ])
