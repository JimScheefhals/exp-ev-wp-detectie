from ev_detection.src.features.instances.base import ModelFeature
from ev_detection.src.features.instances.max import MaxFeature
from ev_detection.src.types.feature_names import FeatureName

feature_types: dict[FeatureName: ModelFeature] = {
    FeatureName.MAX: MaxFeature
}