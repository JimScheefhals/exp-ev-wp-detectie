import pandas as pd
import random

from ev_detection.src.input.baseload_profiles import BaseloadProfiles
from ev_detection.src.input.charging_profile import ChargingProfiles
from ev_detection.src.types.profiles import WeekProfile

charging_profiles = ChargingProfiles()
baseload_profiles = BaseloadProfiles()

class SyntheticProfiles:

    def render_samples(self, n_profiles: int, ratio_positives: float = 0.5) -> list[WeekProfile]:
        """
        :param n_profiles: total number of profiles
        :param ratio_positives: ratio of positive profiles
        :return: list of combined positive and negative synthetic profiles
        """
        samples = (
            self.render_positives(int(n_profiles * ratio_positives)) +
            self.render_negatives(int(n_profiles * (1 - ratio_positives)))
        )
        random.shuffle(samples)
        return samples

    def render_positives(self, n_profiles: int) -> list[WeekProfile]:
        """
        Generate synthetic profiles by combining samples from PULSE baseload profiles with E-laad charging profiles.
        The profiles cover one week and the interval is 15 minutes and contain a charging profile.
        :param n_profiles: The number of profiles to generate
        :return: synthetic smart meter data
        """
        return [
            charging_sample + baseload_sample
            for charging_sample, baseload_sample in zip(
                charging_profiles.sample_weekly_profiles(n_profiles),
                baseload_profiles.sample_weekly_profiles(n_profiles)
            )
        ]

    def render_negatives(self, n_profiles: int) -> list[pd.Series]:
        """
        Generate synthetic profiles by sampling PULSE baseload profiles.
        The profiles cover one week and the interval is 15 minutes and do not contain a charging profile.
        :param n_profiles: The number of profiles to generate
        :return: synthetic smart meter data
        """
        return baseload_profiles.sample_weekly_profiles(n_profiles)

    def get_datetimes(self):
        """
        :return: timestamps for the profiles generated by self.render()
        """
        return charging_profiles.get_datetimes_week()