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
These profiles may be sampled using 
```
ChargingProfiles().sample_profiles(N_profiles)
```