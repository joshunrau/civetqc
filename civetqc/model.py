from civetqc.dataset import Dataset
import pickle
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.neighbors import KNeighborsClassifier
from typing import Union


class BaseModel(Dataset):

    def __init__(self, cutoff_value: int = 1, balanced: bool = False, list_features: Union[None, list] = None) -> None:
        super().__init__(cutoff_value=cutoff_value, balanced=balanced, list_features=list_features)
    
    def __str__(self) -> str:
        self.compute_metrics()
        return (
            f"Accuracy: {round(self.accuracy * 100, 2)}%\n"
            f"Precision: {round(self.precision * 100, 2)}%\n"
            f"Recall: {round(self.recall * 100, 2)}%\n"
        )

    def compute_metrics(self):
        self.accuracy = accuracy_score(self.test.target, self.predicted)
        self.precision = precision_score(self.test.target, self.predicted, pos_label = 0)
        self.recall = recall_score(self.test.target, self.predicted, pos_label = 0)
    
    @staticmethod
    def save(model, filepath):
        with open(filepath, 'wb') as f:
            pickle.dump(model, f)

    @staticmethod
    def load(filepath):
        with open(filepath, 'rb') as f:
            model = pickle.load(f)
        return model


class KNeighborsModel(BaseModel):
    
    def __init__(self, k: int, list_features = None) -> None:
        
        super().__init__(cutoff_value = 1, balanced = True, list_features = list_features)
        self.clf = KNeighborsClassifier(n_neighbors=k)
        self.clf.fit(self.train.features, self.train.target)
        self.predicted = self.clf.predict(self.test.features)


if __name__ == "__main__":
    for k in range(1, 11):
        print(f"\nKNN Model (K={k}):")
        model = KNeighborsModel(k, list_features=["LEFT_SURF_SURF", "RIGHT_SURF_SURF"])
        print(model)
