import pandas as pd
import numpy as np

from ev_detection.src.data_loader import DataLoader
from random import sample

_loader = DataLoader()

TIMESTEP = 1 / 4 # [hours]
LOAD_TO_SOC_FACTOR = TIMESTEP  # kW to kWh per timestep
MINIMUM_SOC_AT_START_SESSION = (
    0.1  # Assumed minimum state of charge, as a fraction of the battery capacity
)

class ChargingProfiles:

    def __init__(self):
        self.all_profiles = _loader.load_elaad_charging_profiles()
        self.unique_run_ids = self.all_profiles["run_id"].unique().tolist()

    def get_datetimes(self):
        return pd.to_datetime(
            self.all_profiles[self.all_profiles["run_id"] == self.unique_run_ids[0]]["date_time"],
            utc=True
        )

    def sample_profiles(self, N_profiles: int = 1) -> list[pd.Series]:
        """
        Randomly sample 'N_profiles' charging profiles
        :param N_profiles: the number of profiles to sample
        :return: list of charging profiles
        """

        selected_run_ids = sample(self.unique_run_ids, N_profiles)
        return [
            self._get_profile_by_id(run_id)
            for run_id in selected_run_ids
        ]

    def _get_profile_by_id(self, run_id: int) -> pd.Series:
        return self.all_profiles[self.all_profiles["run_id"] == run_id]["power"]

    def duration_charging_sessions(self) -> pd.Series:
        """
        Determine the duration of all charging sessions in the available profiles.
        :return: series of durations of charging sessions in hours
        """
        return pd.concat(
            [
                ChargingProfile(self._get_profile_by_id(id)).get_duration_charging_sessions()
                for id in self.unique_run_ids
                if ChargingProfile(self._get_profile_by_id(id)).get_charging_capacity() > 0
            ]
        ).reset_index(drop=True)
        return durations

    def loaded_charge_sessions(self) -> pd.Series:
        """
        Determine the charge loaded into the EV by the charger at the end of every charging session.
        """
        return pd.concat(
            [
                ChargingProfile(self._get_profile_by_id(id)).get_loaded_charge_sessions()
                for id in self.unique_run_ids
                if ChargingProfile(self._get_profile_by_id(id)).get_charging_capacity() > 0
            ]
        ).reset_index(drop=True)

    def start_charging_sessions(self) -> pd.Series:
        """
        Determine the start time of every charging session.
        """
        datetime = self.get_datetimes()
        return pd.Series(
            [
                datetime.iloc[idx].hour
                for id in self.unique_run_ids
                for idx in ChargingProfile(self._get_profile_by_id(id)).get_start_charging_sessions()
            ]
        )


class ChargingProfile:
    def __init__(self, power: pd.Series):
        self.power = power

    def get_charging_capacity(self) -> float:
        return self.power.max()

    def get_battery_capacity(self) -> float:
        """
        Get the battery capacity of the EV.
        Assumptions:
        - The battery has a SoC of MINIMUM_SOC_AT_START_SESSION at the start of each session
        - The battery is at least once fully charged in the profile
        :return:
        """
        return self.get_loaded_charge().max() / (1 - MINIMUM_SOC_AT_START_SESSION)

    def get_duration_charging_sessions(self) -> pd.Series:
        """
        Determine the duration of all charging sessions in the profile.
        :return: series of durations of charging sessions in hours
        """
        # Determine wheter the car is charging (1) or not (0)
        charging = np.zeros(len(self.power))
        charging[self.power > 0] = 1

        # Determine the number of subsequent timesteps the car is charging at every timestep
        n_timesteps_charging = cumulative_sum_with_reset(charging)

        # Find and return the duration of every charging session
        idx_end_session = np.argwhere(n_timesteps_charging[1:] - n_timesteps_charging[:-1] < 0)[:, 0]
        duration_sessions = n_timesteps_charging[idx_end_session] # [timesteps]
        return pd.Series(duration_sessions * TIMESTEP) # [hours]

    def get_loaded_charge_sessions(self):
        """
        Determine the charge loaded into the EV by the charger at the end of every charging session.
        """
        # Find the loaded charge at every timestep
        loaded_charge = cumulative_sum_with_reset(
            self.power.values * LOAD_TO_SOC_FACTOR
        ) # [kWh]

        # Find and return the charge at the end of every charging session
        idx_end_session = np.argwhere(loaded_charge[1:] - loaded_charge[:-1] < 0)[:, 0]
        return pd.Series(loaded_charge[idx_end_session]) # [kWh]

    def get_start_charging_sessions(self):
        """
        Determine the start of every charging session in indices.
        """
        # Determine wheter the car is charging (1) or not (0)
        charging = np.zeros(len(self.power))
        charging[self.power > 0] = 1

        # Find and return the start of every charging session
        idx_start_session = np.argwhere(
            np.logical_and(
                charging[1:] == 1,
                charging[:-1] == 0
            )
        )[:, 0]
        return pd.Series(idx_start_session) # [timesteps]


def cumulative_sum_with_reset(x: np.ndarray) -> np.ndarray:
    """
    Perform a cumulative sum, which resets to 0 when the input x becomes zero.
    """
    not_loading = x == 0
    cumulative_sum = np.cumsum(x)
    diff = np.diff(np.concatenate(([0.0], cumulative_sum[not_loading])))
    x[not_loading] = -diff
    return np.cumsum(x)
