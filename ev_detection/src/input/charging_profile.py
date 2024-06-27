import pandas as pd
import numpy as np


from ev_detection.src.input.data_loader import DataLoader
from ev_detection.src.types.profiles import WeekProfile, YearProfile

_loader = DataLoader()

TIMESTEP = 1 / 4 # [hours]
LOAD_TO_SOC_FACTOR = TIMESTEP  # kW to kWh per timestep
MINIMUM_SOC_AT_START_SESSION = (
    0.1  # Assumed minimum state of charge, as a fraction of the battery capacity
)
CAPACITY_DISTRIBUTION = {
    4: 0.4,
    8: 0.3,
    11: 0.3
} # Assumed distribution of charger capacities

class ChargingProfiles:

    def __init__(self):
        self.all_profiles = _loader.load_elaad_charging_profiles()
        self.id_capacity_mapping =self.all_profiles[
            ["run_id", "capacity"]
        ].drop_duplicates().set_index("run_id", drop=True).to_dict()["capacity"]
        self.unique_run_ids = list(self.id_capacity_mapping.keys())
        self.all_profiles = self._expand_datetime(self.all_profiles)
        self.run_id_week_combinations = self.find_possible_run_id_week_combinations()

    def get_datetimes(self) -> pd.Series:
        return self.all_profiles[self.all_profiles["run_id"] == self.unique_run_ids[0]]["datetime"]

    def sample_yearly_profiles(
            self,
            N_profiles: int = 1,
            capacity_distribution: dict[int, float] = CAPACITY_DISTRIBUTION
    ) -> list[YearProfile]:
        """
        Randomly sample 'N_profiles' yearly charging profiles, where charger capacities are selected according to the
        given distribution.
        :param N_profiles: the number of profiles to sample
        :param capacity_distribution: the distribution of charger capacities
        :return: list of charging profiles
        """
        id_prob_mapping = {k: capacity_distribution[v] for k, v in
                           self.id_capacity_mapping.items()}
        id_prob_mapping = {k: v / sum(id_prob_mapping.values()) for k, v in id_prob_mapping.items()}
        selected_run_ids = np.random.choice(
            list(id_prob_mapping.keys()),
            p=list(id_prob_mapping.values()),
            size=N_profiles
        )
        return [
            self._get_profile_by_id(run_id)
            for run_id in selected_run_ids
        ]

    def get_datetimes_week(self) -> pd.Series:
        df_datetime = self.all_profiles[self.all_profiles["run_id"] == self.unique_run_ids[0]]
        return df_datetime[df_datetime["week"] == 2]["datetime"]

    def sample_weekly_profiles(
        self,
        n_profiles: int = 1,
        capacity_distribution: dict[int, float] = CAPACITY_DISTRIBUTION
    ) -> tuple[list[WeekProfile], pd.DataFrame]:
        """
        Randomly sample 'N_profiles' weekly charging profiles, where charger capacities are selected according to the
        given distribution. Use only combinations of run_id and week_nr for which the profile contains at least one
        (or part of a) charging session.
        :param n_profiles: the number of profiles to sample
        :return: list of charging profiles
        """
        probabilities = np.array([
            capacity_distribution[self.id_capacity_mapping[t[0]]]
            for t in self.run_id_week_combinations
        ])

        # Sample combinations
        selected_combinations = np.array(self.run_id_week_combinations)[
            np.random.choice(
                np.arange(len(self.run_id_week_combinations)),
                p=probabilities / sum(probabilities),
                size=n_profiles
            )
        ]
        meta_data = pd.DataFrame({
            "id": np.arange(n_profiles),
            "charging_id": [run_id for run_id, week in selected_combinations],
            "charging_week": [week for run_id, week in selected_combinations]
        })
        return [
            self.get_profile_by_id_week(run_id, week).reset_index(drop=True)
            for run_id, week in selected_combinations
        ], meta_data

    def find_possible_run_id_week_combinations(self) -> list[tuple[int, int]]:
        """
        Find all combinations of run_id and week number for which the maximum of the corresponding profile is larger
        than 1 and the profile contains the right amount of timesteps.
        """
        grouped = self.all_profiles[["run_id", "week", "power"]].groupby(by=["run_id", "week"])

        grouped_max = grouped.max()
        filtered_max = grouped_max[grouped_max["power"] > 1]

        grouped_count = grouped.count()
        filtered_count = grouped_count[grouped_count["power"] == 7 * 24 * 4]

        return list(set(filtered_max.index).intersection(set(filtered_count.index)))

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

    def start_charging_sessions(self) -> tuple[pd.Series, pd.Series]:
        """
        Determine the start time, in hours and weekdays, of every charging session.
        """
        datetime = self.get_datetimes()
        weekday, hour = zip(*[
            (datetime.iloc[idx].weekday(), datetime.iloc[idx].hour)
            for id in self.unique_run_ids
            for idx in ChargingProfile(self._get_profile_by_id(id)).get_start_charging_sessions()
        ])
        return pd.Series(weekday), pd.Series(hour)

    def get_charger_capacities(self):
        return [
            ChargingProfile(self._get_profile_by_id(id)).get_charging_capacity()
            for id in self.unique_run_ids
        ]

    def _get_profile_by_id(self, run_id: int) -> pd.Series:
        return self.all_profiles[self.all_profiles["run_id"] == run_id]["power"]

    def get_profile_by_id_week(self, profile_id: int, week_nr: int) -> pd.Series:
        yearly_profile = self.all_profiles[self.all_profiles["run_id"] == profile_id]
        return yearly_profile[yearly_profile["week"] == week_nr]["power"]

    def _expand_datetime(self, df: pd.DataFrame):
        df_datetime = pd.DataFrame({
            "date_time": self.all_profiles[self.all_profiles["run_id"] == self.unique_run_ids[0]]["date_time"]
        })
        df_datetime["datetime"] = pd.to_datetime(df_datetime["date_time"], utc=True).dt.tz_convert("Europe/Amsterdam")
        df_datetime["week"] = df_datetime["datetime"].dt.isocalendar().week
        df_datetime["weekday"] = df_datetime["datetime"].dt.weekday
        df_datetime["time"] = df_datetime["datetime"].dt.time
        df_datetime["hour"] = df_datetime["datetime"].dt.hour
        return df.merge(
            df_datetime,
            on="date_time",
            how="inner"
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
        return self.get_loaded_charge_sessions().max() / (1 - MINIMUM_SOC_AT_START_SESSION)

    def get_duration_charging_sessions(self) -> pd.Series:
        """
        Determine the duration of all charging sessions in the profile.
        :return: series of durations of charging sessions in hours
        """
        # Determine whether the car is charging (1) or not (0)
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
        )[:, 0] + 1
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

if __name__ == "__main__":
    charging_profiles = ChargingProfiles()
    samples = charging_profiles.sample_weekly_profiles(10)
