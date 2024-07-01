import pandas as pd
import numpy as np
from scipy.signal import find_peaks

# Parameters for peak_finding
MIN_PROMINENCE = 2 # [kW]
MIN_WIDTH = 1      # [timesteps]
MIN_HEIGHT = 3.7   # [kW]

def peak_detection_single_sample(sample: pd.Series):
    """
    Peak detection algorithm
    :param sample: profile to run peak detection on
    :return: tuple containing the peak indices and properties
    """
    return find_peaks(
        sample,
        prominence=MIN_PROMINENCE,
        width=MIN_WIDTH,
        height=MIN_HEIGHT,
    )

def peak_detection_multiple_samples(samples: dict[int, pd.Series]) -> dict[int, [dict[str, np.ndarray]]]:
    """
    Run the peak detection for multiple samples and write the results to a dictionary
    :param samples: profiles to run the peak detection on, mapped to an id
    :return: dictionary of peak detection results for every sample in samples, mapped to the corresponding id
    """
    properties_all = {}
    for id, sample in samples.items():
        peaks_sample, properties_sample = peak_detection_single_sample(sample)
        properties_sample["idx"] = peaks_sample
        properties_all[id] = properties_sample
    return properties_all