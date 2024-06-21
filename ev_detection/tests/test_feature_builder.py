import pandas as pd
from ev_detection.src.features.feature_builder import FeatureBuilder

def test_feature_builder():
    # GIVEN a list of profiles
    test_profiles = [pd.Series([1, 1, 2, 3, 1, 1, 1, 1]), pd.Series([2, 2, 2, 4, 5, 6, 2, 2])]
    datetime = pd.Series(pd.date_range("2021-01-01", periods=8, freq="H"))

    # WHEN the FeatureBuilder is instantiated, with no features specified
    _builder = FeatureBuilder(all_profiles=test_profiles, datetime=datetime)

    # THEN we expect the feature builder to construct all features from the profiles
    assert _builder._feature_input.all_profiles == test_profiles