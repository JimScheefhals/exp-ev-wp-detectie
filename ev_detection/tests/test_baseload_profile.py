from ev_detection.src.input.baseload_profiles import BaseloadProfiles

import pandas as pd
import numpy as np

N_PROFILES = 1000

def test_baseload_profile():
   # GIVEN a BaseloadProfiles instance
   _baseload = BaseloadProfiles()

   # WHEN generating weekly samples
   result = _baseload.sample_weekly_profiles(n_profiles=N_PROFILES)

   # THEN we expect the result to be a dictionary with the correct number of samples and a dataframe with meta-data
   assert len(result[0]) == N_PROFILES
   assert isinstance(result[0], dict)
   assert len(result[1]) == N_PROFILES
   assert isinstance(result[1], pd.DataFrame)

   # AND we expect the resulting profiles to have the correct length
   assert np.all([len(profile) == 7 * 24 * 4 for profile in result[0].values()])



