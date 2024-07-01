import pandas as pd

from ev_detection.src.features.feature_builder import FeatureBuilder
from ev_detection.src.input.load_profiles import LoadProfiles
from ev_detection.src.types.classifier_names import ClassifierName
from ev_detection.src.types.feature_names import FeatureName
from ev_detection.src.types.classifier_types import classifier_types
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
)

class ClassifierModel:

    def __init__(
            self,
            all_profiles: dict[int, pd.Series],
            datetime: pd.Series,
            meta_data: pd.DataFrame(),
            _features: list[FeatureName] = FeatureName.__members__.values()
    ):

        self._feature_builder = FeatureBuilder(
            all_profiles=all_profiles,
            datetime=datetime,
            meta_data=meta_data,
            _features=_features
        )
        self._feature_builder.build()
        self.train_test_split()

    def train_test_split(self, random_state: int = 42):

        # Get the features and labels
        features = self._feature_builder.get_features()
        labels = self._feature_builder.get_labels()

        # Split the data into a training and test set
        self.features_train, self.features_test, self.labels_train, self.labels_test = train_test_split(
            features, labels, test_size=0.2, random_state=random_state
        )

    def train_and_predict(self, classifier: ClassifierName) -> pd.Series:
        """
        Predict the labels of the test set using the given classifier.
        """
        model = classifier_types[classifier]
        model.fit(self.features_train, self.labels_train)
        self.labels_predict = model.predict(self.features_test)

    def get_predictions(self, only_wrong_predictions: bool = False) -> pd.Series:
        """
        Get the results of the last prediction.
        When only_wrong_predictions is True, only the wrong predictions are returned.
        """
        result = self.features_test.copy()
        result["label"] = self.labels_test
        result["prediction"] = self.labels_predict
        result = result.drop("label").merge(self._feature_builder.get_meta_data(), on="id", how="inner")
        if only_wrong_predictions:
            return result[result["label"] != result["prediction"]]
        else:
            return result

    def get_metrics(self) -> dict[str, float | int | pd.DataFrame]:
        """
        Get the metrics of the last prediction.
        """
        return {
            "accuracy": accuracy_score(self.labels_test, self.labels_predict),
            "confusion_matrix": confusion_matrix(self.labels_test, self.labels_predict),
            "f1_score": f1_score(self.labels_test, self.labels_predict),
            "precision": precision_score(self.labels_test, self.labels_predict),
            "recall": recall_score(self.labels_test, self.labels_predict),
        }

if __name__ == "__main__":
    load_profiles = LoadProfiles()
    samples, meta_data = load_profiles.render_samples(10)
    datetime = load_profiles.get_datetimes()
    classifier_model = ClassifierModel(samples, datetime, meta_data)
    classifier_model.train_and_predict(ClassifierName.LOGREGRESSION)
    metrics = classifier_model.get_metrics()
    print(metrics)
