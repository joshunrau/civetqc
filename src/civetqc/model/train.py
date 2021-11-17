import pickle

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

from ..data.create_dataset import Dataset, MergedDataset


class TrainModel(Dataset):

    def __init__(self, data: MergedDataset) -> None:
        super().__init__(data)
        self.clf = RandomForestClassifier(n_estimators=100, random_state=0)
        self.clf.fit(self.train.features, self.train.target)
        self.predicted = self.clf.predict(self.test.features)

    def __str__(self) -> str:
        return classification_report(self.test.target, self.predicted)

    def save(self, filepath):
        with open(filepath, 'wb') as f:
            pickle.dump(self.clf, f)
