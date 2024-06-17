import pandas as pd
import numpy as np
from ev_detection.src.data_loader import DataLoader

loader = DataLoader()

MEDIAN_SJV = 2000 # sjv [kwh], assumed median, calculated from a set of 17 LV grids
STD_SJV = 1000 # assumed standard deviation for the sjv, rough estimate
SJV_TO_AVG_KW = 4 / 52 # factor for converting sjv to an average power in kW for a week

class BaseloadProfiles:

    def sample_weekly_profiles(self, n_profiles: int) -> list[pd.Series]:
        """
        Sample dummy baseload profiles. This is to be replaced with PULSE smart meter data.
        The baseloads are assumed to represent the consumed power of a E1 or E2 customer, which does not charge an EV
        at the considered connection. The profile covers a week and the interval is 15 minutes.
        :param n_profiles:
        :return: weekly baseload profiles
        """
        yearly_consumption = np.random.normal(loc=MEDIAN_SJV, scale=STD_SJV, size=n_profiles)
        dummy_profiles = [
            pd.Series(np.clip(dummy_profile(),
                a_min=0,
                a_max=None
            ))
            for i in range(n_profiles)
        ]
        return [
            profile / profile.sum() * (sjv * SJV_TO_AVG_KW)
            for profile, sjv in zip(dummy_profiles, yearly_consumption)
        ]

def dummy_profile() -> np.ndarray:
    return (
            1
            + 2 * np.sin((np.arange(0, 7 * 24 * 4, 1) / (12 * 4) - 0.3) * (2*np.pi))
            + np.random.randn(7 * 24 * 4)
    )