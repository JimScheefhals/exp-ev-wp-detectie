import pandas as pd

from ev_detection.src.features.feature_input import FeatureInput
from ev_detection.src.input.synthetic_profiles import SyntheticProfiles
from ev_detection.src.types.feature_names import FeatureName
from ev_detection.src.types.feature_types import feature_types

class FeatureBuilder:
    """
    This class is used to construct features from the original data.
    """

    def __init__(
            self,
            all_profiles: dict[int, pd.Series],
            datetime: pd.Series,
            meta_data: pd.DataFrame(),
            _features: list[FeatureName] = FeatureName.__members__.values()
    ):
        self._feature_input = FeatureInput(
            all_profiles=all_profiles,
            datetime=datetime
        )
        self._features = _features
        self.meta_data = meta_data

    def build(self) -> dict[FeatureName: pd.Series]:
        """ Build features and write them to a DataFrame."""
        self.res = self.meta_data.copy()
        for feature_name in self._features:
            feature = feature_types[feature_name]
            self.res[feature_name.value] = self.res["id"].map(
                feature(self._feature_input).get()
            )

    def get_features(self) -> dict[FeatureName: pd.Series]:
        return self.res

    def get_input(self) -> FeatureInput:
        return self._feature_input

syn_profiles = SyntheticProfiles()
samples, meta_data = syn_profiles.render_samples(10)
datetime = SyntheticProfiles().get_datetimes()
builder = FeatureBuilder(samples, datetime, meta_data)
builder.build()
features = builder.get_features()
features