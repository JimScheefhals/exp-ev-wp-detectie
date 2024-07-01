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
from ev_detection.src.input.charging_profile import ChargingProfiles()

samples = ChargingProfiles().sample_profiles(N_profiles)
```
The corresponding datetimes are extracted via 
```
ChargingProfiles().get_datetimes()
```

The same may be done for weekly data:
```
samples = ChargingProfile().sample_weekly_profiles()
ChargingProfile().get_datetimes_week()
```


## Sampling synthetic smart meter data
Synthetic smart meter data may be used to train a supervised learning model. 
Positives (i.e. has a home charger for EV) are constructed by combining a 'clean' baseload, 
with a charging profile as described above. 
Negatives are just copies of the baseload profiles.
In this case, weekly profiles are the default.
```
from ev_detection.src.input.synthetic_profiles import LoadProfiles

# sample positives
samples = LoadProfiles().render_positives(n_profiles)

# sample negatives
samples = LoadProfiles().render_negatives(n_profiles)

# sample a mixture of positives and negatives
samples, meta_data = LoadProfiles().render_samples(n_profiles, ratio_positives)

# get datetimes
LoadProfiles().get_datetimes()
```

## Building features from the data
For feature building we use the `FeatureBuilder` interface which calls
a number of specified features. Using the `build()` method, the features are
constructed and stored in a pandas DataFrame which can be exported using `get_features()`.
```
from ev_detection.src.features.feature_builder import FeatureBuilder
_builder = FeatureBuilder(samples, meta_data)
_builder.build()
features = _builder.get_features()
```

## Training a model
We use the constructed features to train a model. The model is trained using the `ModelTrainer` interface.
This interface can be used together with a number of different Classifiers, which are specified in 
the `classifier_types` parameter. We may perform a test of the model using the `test_prediction()` method and 
get the metrics using the `get_metrics()` method.
```
from ev_detection.src.model.model_trainer import ModelTrainer
classifier_model = ClassifierModel(samples, datetime, meta_data)
classifier_model.test_prediction(ClassifierName.LOGREGRESSION)
classifier_model.get_metrics()
```