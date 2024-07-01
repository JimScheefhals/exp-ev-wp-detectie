import pandas as pd

from ev_detection.src.input.data_loader import DataLoader

_loader = DataLoader()

def test_dataloader():
    # GIVEN a DataLoader instance
    # WHEN the DataLoader is instantiated

    # THEN we expect the DataLoader to be able to load at least 10 elaad charging profiles
    result_charging = _loader.load_elaad_charging_profiles()
    assert result_charging["run_id"].unique().shape[0] > 10
    assert isinstance(result_charging, pd.DataFrame)


    # AND we expect the DataLoader to be able to load at least 10 pulse baseload profiles
    result_pulse = _loader.load_pulse_baseload_profiles()
    assert isinstance(result_pulse, pd.DataFrame)
    assert result_pulse.columns.shape[0] > 10

    # AND we expect the DataLoader to be able to load pulse labels, where there is a minimum of 10 negative labels
    result_labels = _loader.load_pulse_labels()
    assert isinstance(result_labels, pd.DataFrame)
    assert len(result_labels[result_labels["label"] == 0]) > 10
