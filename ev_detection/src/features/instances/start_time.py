from ev_detection.src.features.instances.base import ModelFeature
from ev_detection.src.types.feature_names import FeatureName

import numpy as np

class MeanStartTimeFeature(ModelFeature):
    """
    Find the average start time of the plateaus found by the PlateauDetection
    """

    @property
    def type(self) -> FeatureName:
        return FeatureName.MEAN_START_TIME

    def get(self) -> dict[int, float]:
        return {
            id: np.nanmean([
                self.input.datetime.iloc[plateau_idx[0]].hour + self.input.datetime.iloc[plateau_idx[0]].minute / 60
                for plateau_idx in plateaus
                ]) if len(plateaus) > 0 else 0
            for id, plateaus in self.input.plateaus.items()
        }

class ConsistentStartTime(ModelFeature):
    """
    Find the standard deviation of the start time of the plateaus found by the PlateauDetection.
    The lower this value is, the more consistent is the start time of the plateau. Extremely consistent plateaus
    generally belong to profiles of shops with regular opening hours, etc.
    """

    @property
    def type(self) -> FeatureName:
        return FeatureName.CONSISTENT_START_TIME

    def get(self) -> dict[int, float]:
        return {
            id: np.std([
                self.input.datetime.iloc[plateau_idx[0]].hour + self.input.datetime.iloc[plateau_idx[0]].minute / 60
                for plateau_idx in plateaus
                ]) if len(plateaus) > 0 else 0
            for id, plateaus in self.input.plateaus.items()
        }