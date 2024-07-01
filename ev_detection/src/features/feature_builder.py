import pandas as pd

from ev_detection.src.features.feature_input import FeatureInput
from ev_detection.src.input.load_profiles import LoadProfiles
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
            datetime=datetime,
            meta_data=meta_data
        )
        self._features = _features

    def build(self) -> dict[FeatureName: pd.Series]:
        """ Build features and write them to a DataFrame."""
        self.res = self._feature_input.meta_data.copy()
        for feature_name in self._features:
            feature = feature_types[feature_name]
            self.res[feature_name.value] = self.res["id"].map(
                feature(self._feature_input).get()
            )

    def get_features(self) -> pd.DataFrame:
        return self.res[["id"] + [feature.value for feature in self._features]].set_index("id")

    def get_labels(self) -> pd.Series:
        return self._feature_input.meta_data["label"]

    def get_meta_data(self) -> pd.DataFrame:
        return self._feature_input.meta_data

    def get_input(self) -> FeatureInput:
        return self._feature_input


if __name__ == "__main__":
    load_profiles = LoadProfiles()
    samples, meta_data = load_profiles.render_samples(10)
    datetime = load_profiles.get_datetimes()
    builder = FeatureBuilder(samples, datetime, meta_data)
    builder.build()
    features = builder.get_features()
