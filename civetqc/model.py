import numpy as np

from abc import ABC, abstractmethod
from sklearn.metrics import fbeta_score, make_scorer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .dataset import Dataset


class AbstractModel(ABC):

    f2_scorer = make_scorer(fbeta_score, beta=2)

    def __init__(self, data: Dataset) -> None:
        self.data = data
        self.preprocessing = Pipeline([("scaler", StandardScaler())])
        self.data.train["Features"] = self.preprocessing.fit_transform(
            self.data.train["Features"], self.data.train["Target"]
        )
        self.data.test["Features"] = self.preprocessing.transform(
            self.data.test["Features"]
        )

    @abstractmethod
    def fit(self):
        pass

    @property
    @abstractmethod
    def params(self):
        pass

    @property
    @abstractmethod
    def predicted(self):
        pass


class BaseModel(AbstractModel):
    def __init__(self, data: Dataset) -> None:
        super().__init__(data)

    def fit(self):
        return super().fit()

    @property
    def predicted(self):
        return np.zeros(len(self.data.test["Target"]), dtype=np.int64)

    @property
    def params(self):
        return super().params


class KNN(AbstractModel):
    def __init__(self, data: Dataset) -> None:
        super().__init__(data)
