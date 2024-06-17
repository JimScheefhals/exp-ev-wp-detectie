import pandas as pd

TIMESTEPS_IN_HOUR = 4

"""
WeekProfile is a series representing a week
"""
WeekProfile = pd.Series
NUM_TIMESTEPS_WEEK = TIMESTEPS_IN_HOUR * 24 * 7


"""
YearProfile is a series representing a year
"""
YearProfile = pd.Series