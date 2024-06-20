from dataclasses import dataclass
import pandas as pd

@dataclass
class FeatureInput:
    """
    This class is used to store the input data for the construction of features.
    """
    all_profiles = list[pd.Series]