import pandas as pd

from ev_detection.src.features.instances.base import ModelFeature
from ev_detection.src.types.feature_names import FeatureName


class MaxFeature(ModelFeature):

    @property
    def type(self) -> FeatureName:
        return FeatureName.MAX

    def get(self) -> dict[int, float]:
        return {
            id: profile.max()
            for id, profile in self.input.all_profiles.items()
        }