from ev_detection.src.features.instances.base import ModelFeature
from ev_detection.src.features.instances.max import MaxFeature
from ev_detection.src.features.instances.peak_prominence import MaxPeakProminenceFeature, MeanPeakProminenceFeature
from ev_detection.src.features.instances.peak_widths import MaxPeakWidth, MeanPeakWidth
from ev_detection.src.features.instances.start_time import MeanStartTimeFeature
from ev_detection.src.types.feature_names import FeatureName

feature_types: dict[FeatureName: ModelFeature] = {
    FeatureName.MAX: MaxFeature,
    FeatureName.MAX_PEAK_PROMINENCE: MaxPeakProminenceFeature,
    FeatureName.MEAN_PEAK_PROMINENCE: MeanPeakProminenceFeature,
    FeatureName.MAX_PEAK_WIDTH: MaxPeakWidth,
    FeatureName.MEAN_PEAK_WIDTH: MeanPeakWidth,
    FeatureName.MEAN_START_TIME: MeanStartTimeFeature,
}