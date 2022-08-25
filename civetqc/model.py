from __future__ import annotations

import pickle

from pkg_resources import resource_filename

import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.utils.validation import check_is_fitted


class Model:

    resource_path = resource_filename(__name__, "resources/model.pkl")

    def __init__(self, clf: RandomForestClassifier, default_threshold: float) -> None:
        check_is_fitted(clf)
        self.clf = clf
        self.default_threshold = default_threshold

    def predict(self, data: np.ndarray, threshold: float | None = None) -> np.ndarray:
        if threshold is None:
            threshold = self.default_threshold
        return np.where(self.predict_probabilities(data)[:, 1] > threshold, 1, 0)

    def predict_probabilities(self, data: np.ndarray) -> np.ndarray:
        return self.clf.predict_proba(data)

    def save(self) -> None:
        with open(self.resource_path, "wb") as file:
            pickle.dump(self, file)

    @classmethod
    def load(cls) -> Model:
        with open(cls.resource_path, "rb") as file:
            return pickle.load(file)

    @classmethod
    def get_default_threshold(cls) -> float:
        model = Model.load()
        return model.default_threshold
