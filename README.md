# EV and Heatpump detection

This project is a proof of concept, aiming to detect EV charging and/or heatpump profiles from weekly smart meter profile data.

## Getting started
Ensure you have all required files in a local directory
and set up environment variables in a local `.env` file (see `.env.example` for a template). Then install all dependencies using the following command in the terminal:
```shell
poetry install
```

## Sampling charging profiles
Charging profiles are generated using the E-laad LV Profile Generator: 
https://platform.elaad.io/analyse/low-voltage-charging-profiles/. 
Once the resulting csv-files are in "data_dir/elaad/laadprofielen", these profiles may be sampled using:
```
from ev_detection.src.charging_profile import ChargingProfiles()

ChargingProfiles().sample_profiles(N_profiles)
```
The corresponding datetimes are extracted via 
```
ChargingProfiles().get_datetimes()
```

The same may be done for weekly data:
```
ChargingProfile().sample_weekly_profiles()
ChargingProfile().get_datetimes_week()
```


## Sampling synthetic smart meter data
Synthetic smart meter data may be used to train a supervised learning model. 
Positives (i.e. has a home charger for EV) are constructed by combining a 'clean' baseload, 
with a charging profile as described above. 
Negatives are just copies of the baseload profiles.
In this case, weekly profiles are the default.
```
from ev_detection.src.synthetic_profiles import SyntheticProfiles

SyntheticProfiles().render_positives()
SyntheticProfiles().render_negatives()
SyntheticProfiles().get_datetimes()
```