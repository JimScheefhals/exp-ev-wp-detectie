from ev_detection.src.features.instances.base import ModelFeature
from ev_detection.src.types.feature_names import FeatureName

import numpy as np

class MeanStartTimeFeature(ModelFeature):

    @property
    def type(self) -> FeatureName:
        return FeatureName.START_TIME

    def get(self) -> dict[int, float]:
        return {
            id: np.nanmean([
                self.input.datetime.iloc[plateau_idx[0]].hour + self.input.datetime.iloc[plateau_idx[0]].minute / 60
                for plateau_idx in plateaus
                ])
            for id, plateaus in self.input.plateaus.items()
        }
