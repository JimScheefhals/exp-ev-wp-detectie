from ev_detection.src.types.classifier_names import ClassifierName
from sklearn.base import BaseEstimator
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

classifier_types: dict[ClassifierName: BaseEstimator] = {
    ClassifierName.GAUSSIANNB: GaussianNB(),
    ClassifierName.LOGREGRESSION: LogisticRegression(),
    ClassifierName.XGBOOST: XGBClassifier(),
}