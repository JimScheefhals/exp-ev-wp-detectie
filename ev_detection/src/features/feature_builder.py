import pandas as pd
import scipy

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
            all_profiles: list[pd.Series],
            _features: list[FeatureName] = FeatureName.__members__.values()
    ):
        self._feature_input = FeatureInput(
            all_profiles=all_profiles
        )
        self._features = _features

    def build(self) -> dict[FeatureName: pd.Series]:
        res: dict[FeatureName] = {}
        for feature_name in self._features:
            feature = feature_types[feature_name]
            res[feature_name.value] = feature(self._feature_input).get()
        return res


samples = SyntheticProfiles().render_samples(10)
FeatureBuilder(samples).build()