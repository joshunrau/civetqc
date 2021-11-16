from abc import ABC, abstractmethod
from .dataset import Dataset
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import pickle
import numpy as np


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


class BaseModel(ABC):

    """

    Attributes
    ----------
    feature_names : list
        List of the names of the features to be included in the model
    target_names: list
        List of names for the classes
    features: np.ndarray
        2D array of features included in the model
    target: np.ndarray
        1D array of targets
    target_names: list
        List of names for the classes
    train: DataPartition
        Training set
    test: DataPartition
        Testing set
    
    """

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
    @abstractmethod
    def model(self):
        pass

    @property
    def predicted(self):
        return self.model.predict(self.test.features)


class KNeighborsModel(BaseModel):

    def __init__(self, data: Dataset) -> None:
        super().__init__(data)

    @property
    def model(self):
        model = KNeighborsClassifier(n_neighbors=5)


class RandomForestModel(BaseModel):

    def __init__(self, data: Dataset) -> None:
        super().__init__(data)
    
    @property
    def model(self):
        model = RandomForestClassifier(n_estimators=100, random_state=0)
        model.fit(self.train.features, self.train.target)
        return model

def main():
    data = Dataset(cutoff_value=1, balanced=True, list_features=None)
    forest = RandomForestModel(data)
    forest.save("/Users/joshua/Developer/civetqc/data/models/forest.pickle")