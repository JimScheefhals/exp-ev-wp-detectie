from enum import Enum

class FeatureName(Enum):
    MAX = "maximum"
    MAX_PEAK_PROMINENCE = "max_peak_prominence"
    MEAN_PEAK_PROMINENCE = "mean_peak_prominence"
    MAX_PEAK_WIDTH = "max_peak_width"
    MEAN_PEAK_WIDTH = "mean_peak_width"