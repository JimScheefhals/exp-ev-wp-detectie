import numpy as np
import pandas as pd

from ev_detection.src.input.data_loader import DataLoader
from ev_detection.src.types.profiles import WeekProfile, YearProfile

loader = DataLoader()
WATT_TO_KW = 1e-3

class BaseloadProfiles:

    def __init__(self):
        self.all_profiles = loader.load_pulse_baseload_profiles()
        self.profile_ids = self.all_profiles.columns
        self.profile_ids_negative = self._get_ids_with_negative_label()
        self.datetime_utc = pd.date_range(
            start='2023-01-01 00:00', end='2023-12-31 23:45', freq='15min'
        ).to_frame(index=False, name='datetime')["datetime"]
        self.datetime_cet = self.datetime_utc.dt.tz_localize('UTC').dt.tz_convert('Europe/Amsterdam')
        self.week_nr = self.datetime_cet.dt.isocalendar().week

    def sample_weekly_profiles(
            self, n_profiles: int
    ) -> tuple[dict[int, WeekProfile], pd.DataFrame]:
        """
        Sample PULSE baseload profiles with label 0.
        These baseloads are assumed to represent the consumed power of a E1 or E2 customer, which does not charge an EV
        at the considered connection. The profile covers a week and the interval is 15 minutes.
        :param n_profiles:
        :return: weekly baseload profiles
        """
        week_id_combinations = self.find_id_week_combinations()
        selected_combinations = week_id_combinations[
            np.random.choice(
                np.arange(len(week_id_combinations)),
                n_profiles
            )
        ]
        meta_data = pd.DataFrame(selected_combinations, columns=["PULSE_id", "PULSE_week"])
        meta_data["id"] = np.arange(n_profiles)
        return self._get_profiles_by_id_week(selected_combinations), meta_data


    def sample_yearly_profiles(self, n_profiles: int) -> dict[int, YearProfile]:
        """
        Sample PULSE baseload profiles with label 0.
        These baseloads are assumed to represent the consumed power of a E1 or E2 customer, which does not charge an EV
        at the considered connection. The profile covers a year and the interval is 15 minutes.
        :param n_profiles:
        :return: yearly baseload profiles
        """
        profile_ids = np.random.choice(
            self.profile_ids_negative,
            n_profiles
        )
        return self._get_profiles_by_id(profile_ids)

    def _get_profiles_by_id_week(self, week_id_combinations: np.ndarray[tuple[int, str]]) -> dict[int, WeekProfile]:
        return {
            i: self.get_profile_by_id_week(profile_id, week_nr)
            for i, (profile_id, week_nr) in enumerate(week_id_combinations)
        }

    def get_profile_by_id_week(self, profile_id: int, week_nr: int):
        return self.all_profiles[self.week_nr == int(week_nr)][profile_id] * WATT_TO_KW

    def _get_profiles_by_id(self, profile_ids: list[int]) -> dict[int, YearProfile]:
        return {profile_id: self.all_profiles[profile_id] * WATT_TO_KW for profile_id in profile_ids}

    def _get_ids_with_negative_label(self) -> list[int]:
        labels = loader.load_pulse_labels()
        return labels[labels["label"] == 0]["id"].tolist()

    def find_id_week_combinations(self) -> np.ndarray[tuple[int, int]]:
        """
        Find all possible combinations of profile_id and week_nr for negatively labeled profiles and complete weeks.
        """
        week_nrs = self.week_nr.unique()
        week_nrs = week_nrs[week_nrs != 1 | week_nrs != 52] # remove incomplete week
        return np.array([
            (profile_id, week_nr)
            for profile_id in self.profile_ids_negative
            for week_nr in week_nrs
        ])

    def get_datetimes(self) -> pd.Series:
        return self.datetime_cet


if __name__ == "__main__":
    baseload_profiles = BaseloadProfiles()
    profiles = baseload_profiles.sample_weekly_profiles(10)
    profiles
