import pandas as pd

from ev_detection.src.data_loader import DataLoader
from random import sample

_loader = DataLoader()

class ChargingProfiles:

    def __init__(self):
        self.all_profiles = _loader.load_elaad_charging_profiles()
        self.unique_run_ids = self.all_profiles["run_id"].unique().tolist()

    def get_datetimes(self):
        return self.all_profiles[self.all_profiles["run_id"] == self.unique_run_ids[0]]["date_time"]

    def sample_profiles(self, N_profiles: int = 1) -> list[pd.Series]:
        """
        Randomly sample 'N_profiles' charging profiles
        :param N_profiles: the number of profiles to sample
        :return: list of charging profiles
        """

        selected_run_ids = sample(self.unique_run_ids, N_profiles)
        return [
            self.all_profiles[self.all_profiles["run_id"] == run_id]["power"]
            for run_id in selected_run_ids
        ]
