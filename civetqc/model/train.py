import pickle

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

from ..data import make_dataset


class DataPartition:
    """ container for test and training sets """

    def __init__(self, features: np.ndarray, target: np.ndarray) -> None:
        assert isinstance(features, np.ndarray) and isinstance(target, np.ndarray)
        assert features.ndim == 2 and target.ndim == 1
        self.features, self.target = features, target

    def __str__(self) -> str:
        return "Target Class Counts\n" + '\n'.join(
            f"{': '.join([str(y) for y in list(x)])} ({round(x[-1] / len(self.target) * 100, 2)}%)" for x in
            np.array(np.unique(self.target, return_counts=True)).T)


class Model:

    target_names = ["Acceptable", "Unacceptable"]

    def __init__(self, data: Dataset) -> None:
        self.feature_names = data.feature_names
        self.features = data.df[data.feature_names].to_numpy()
        self.target = data.df[data.qcvar].to_numpy()
        x_train, x_test, y_train, y_test = train_test_split(self.features, self.target, random_state=0)
        self.train = DataPartition(x_train, y_train)
        self.test = DataPartition(x_test, y_test)

    def __str__(self) -> str:
        return classification_report(self.test.target, self.predicted)

    def save(self, filepath):
        with open(filepath, 'wb') as f:
            pickle.dump(self.model, f)

    @property
    def model(self):
        model = RandomForestClassifier(n_estimators=100, random_state=0)
        model.fit(self.train.features, self.train.target)
        return model

    @property
    def predicted(self):
        return self.model.predict(self.test.features)
