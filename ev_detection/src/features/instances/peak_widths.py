import numpy as np

from ev_detection.src.features.instances.base import ModelFeature
from ev_detection.src.types.feature_names import FeatureName


class MaxPeakWidth(ModelFeature):

    @property
    def type(self) -> FeatureName:
        return FeatureName.MAX_PEAK_WIDTH

    def get(self) -> dict[int, float]:
        return {
            id: prop["widths"].max(initial=0)
            for id, prop in self.input.peak_properties.items()
        }


class MeanPeakWidth(ModelFeature):

    @property
    def type(self) -> FeatureName:
        return FeatureName.MEAN_PEAK_WIDTH

    def get(self) -> dict[int, float]:
        return {
            id: np.append(prop["widths"], 0).mean()
            for id, prop in self.input.peak_properties.items()
        }
