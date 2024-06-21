import numpy as np

from ev_detection.src.features.instances.base import ModelFeature
from ev_detection.src.types.feature_names import FeatureName


class MaxPeakProminenceFeature(ModelFeature):

    @property
    def type(self) -> FeatureName:
        return FeatureName.MAX_PEAK_PROMINENCE

    def get(self) -> dict[int, float]:
        return {
            id: prop["prominences"].max(initial=0)
            for id, prop in self.input.peak_properties.items()
        }


class MeanPeakProminenceFeature(ModelFeature):

    @property
    def type(self) -> FeatureName:
        return FeatureName.MEAN_PEAK_PROMINENCE

    def get(self) -> dict[int, float]:
        return {
            id: np.append(prop["prominences"], 0).mean()
            for id, prop in self.input.peak_properties.items()
        }
