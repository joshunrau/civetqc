from __future__ import annotations

import pickle

from pkg_resources import resource_filename

import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.utils.validation import check_is_fitted


class Model:

    resource_path = resource_filename(__name__, "resources/model.pkl")

    def __init__(self, clf: RandomForestClassifier, threshold: float) -> None:
        check_is_fitted(clf)
        self.clf = clf
        self.threshold = threshold

    def predict(self, data: np.ndarray, labels: dict = None) -> np.ndarray:
        if labels is None:
            return np.where(self.clf.predict_proba(data)[:, 1] > self.threshold, 1, 0)
        return np.where(self.clf.predict_proba(data)[:, 1] > self.threshold, labels[1], labels[0])

    def save(self) -> None:
        with open(self.resource_path, "wb") as file:
            pickle.dump(self, file)

    @classmethod
    def load(cls) -> Model:
        with open(cls.resource_path, "rb") as file:
            return pickle.load(file)
