from .dataset import Dataset
import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import precision_score, recall_score, f1_score


class Model:

    def __init__(self, clf, target_test, target_features, target_predicted) -> None:
        self.clf = clf
        self.pred = self.clf.predict(target_features)
        self.precision_score = precision_score(target_test, target_predicted, average='binary')
        self.recall_score = recall_score(target_test, target_predicted, average='binary')
        self.f1_score = f1 = f1_score(target_test, target_predicted, average='binary')

    def __str__(self) -> str:
        return (
            f"Precision: {round(self.precision_score, 2)}"
            f"Recall: {round(self.recall_score, 2)}"
            f"F1: {round(self.f1_score, 2)}"
        )

    @staticmethod
    def save_model(model, filepath):
        with open(filepath, 'wb') as f:
            pickle.dump(model, f)

    @staticmethod
    def load_model(filepath):
        with open(filepath, 'rb') as f:
            model = pickle.load(f)
        return model


class Modeler(Dataset):

    def __init__(self, civet_csv: str, user_csv: str, cutoff_value: int = 1) -> None:
        super().__init__(civet_csv, user_csv, cutoff_value)
        self.logistic_regression = Model(LogisticRegression().fit(self.features.train, self.target.train))
        self.knn = Model(KNeighborsClassifier(n_neighbors=6))




        
