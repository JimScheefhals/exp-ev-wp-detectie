import pandas as pd
from ev_detection.src.features.feature_builder import FeatureBuilder
from ev_detection.src.types.feature_names import FeatureName


def test_feature_builder_input():
    # GIVEN a list of profiles
    test_profiles = {
        0: pd.Series([1, 1, 2, 3, 1, 1, 1, 1]),
        1: pd.Series([2, 2, 2, 4, 5, 6, 2, 2])
    }
    datetime = pd.Series(pd.date_range("2021-01-01", periods=8, freq="H"))
    meta_data = pd.DataFrame({"id": [0, 1], "type": ["test1", "test2"]})

    # WHEN the FeatureBuilder is instantiated, with no features specified
    _builder = FeatureBuilder(
        all_profiles=test_profiles,
        datetime=datetime,
        meta_data=meta_data,
    )

    # THEN we expect the feature builder to contain the feature input for feature construction
    assert _builder.get_input().all_profiles == test_profiles
    assert _builder.get_input().datetime.equals(datetime)
    assert _builder.get_input().meta_data.equals(meta_data)

def test_feature_builder_build():
    # GIVEN a list of profiles
    test_profiles = {
        0: pd.Series([1, 1, 2, 3, 1, 1, 1, 1]),
        1: pd.Series([2, 2, 2, 4, 5, 6, 2, 2])
    }
    datetime = pd.Series(pd.date_range("2021-01-01", periods=8, freq="H"))
    meta_data = pd.DataFrame({"id": [0, 1], "type": ["test1", "test2"]})

    # WHEN the FeatureBuilder is instantiated, with a specified feature
    _builder = FeatureBuilder(
        all_profiles=test_profiles,
        datetime=datetime,
        meta_data=meta_data,
        _features=[FeatureName.MAX]
    )
    _builder.build()

    # THEN we expect the feature builder to obtain the correct values for the specified feature
    assert _builder.get_features().shape[1] == 3
    assert _builder.get_features()["maximum"].to_list() == [3, 6]