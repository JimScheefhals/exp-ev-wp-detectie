import pandas as pd
import numpy as np

from ev_detection.src.input.charging_profile import ChargingProfile, ChargingProfiles

N_PROFILES = 1000

def test_charging_profiles():
    # GIVEN a ChargingProfiles instance
    _charging = ChargingProfiles()

    # WHEN generating weekly samples
    result = _charging.sample_weekly_profiles(n_profiles=N_PROFILES)

    # THEN we expect the result to be a list with the correct number of samples and a dataframe with meta-data
    assert len(result[0]) == N_PROFILES
    assert isinstance(result[0], list)
    assert len(result[1]) == N_PROFILES
    assert isinstance(result[1], pd.DataFrame)

    # AND we expect the resulting profiles to have the correct length
    assert np.all([len(profile) == 7 * 24 * 4 for profile in result[0]])

    # AND we expect the resulting profiles to at least contain one timestep at which an EV is charging
    assert np.all([profile.sum() > 0 for profile in result[0]])


def test_charging_profile():
    # GIVEN a profile
    test_profile = pd.Series([0, 0, 1, 1, 2, 1, 1.5, 0, 0, 0, 0, 0, 3, 3, 3, 3, 3, 3, 0, 0, 0, 0, 0, 0])

    # WHEN creating the ChargingProfile instance
    _charging = ChargingProfile(test_profile)

    # THEN we expect the ChargingProfile to have the correct charging capacity
    assert _charging.get_charging_capacity() == 3

    # AND we expect the ChargingProfile to have the correct battery capacity (slightly higher than the sum of the
    # longest charging periods didived by the timestep length)
    assert _charging.get_battery_capacity() > 6 * 3 / 4

    # AND we expect the ChargingProfile to have the correct start indices of charging sessions
    assert _charging.get_start_charging_sessions().to_list() == [2, 12]

