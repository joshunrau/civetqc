from .dataset import Dataset
import numpy as np
import pickle
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import precision_score, recall_score, f1_score

class Model:

    def __init__(self, dataset: Dataset) -> None:
        self.dataset = dataset
    
    def __str__(self) -> str:
        return (
            "----------------------------------------------------------------------\n"
            "QC Value Percentages:\n"
            f"0: {round(sum(self.dataset.df['QC'] == 0) / len(self.dataset.df) * 100, 2)}%\n"
            f"1: {round(sum(self.dataset.df['QC'] == 1) / len(self.dataset.df) * 100, 2)}%\n"
            "----------------------------------------------------------------------\n"
        )
    
    def knn(self):
        knn = KNeighborsClassifier(n_neighbors=6)
        knn.fit(self.dataset.feat_train, self.dataset.targ_train)
        predicted_target = knn.predict(self.dataset.feat_test)

        precision = precision_score(self.dataset.targ_test, predicted_target, average='binary')
        print(f"Precision: {round(precision, 2)}")

        recall = recall_score(self.dataset.targ_test, predicted_target, average='binary')
        print(f"Recall: {round(recall, 2)}")

        f1 = f1_score(self.dataset.targ_test, predicted_target, average='binary')
        print(f"F1: {round(f1, 2)}")

    @staticmethod
    def save_model(model, filepath):
        with open(filepath, 'wb') as f:
            pickle.dump(model, f)
    @staticmethod
    def load_model(filepath):
        with open(filepath, 'rb') as f:
            model = pickle.load(f)
        return model



        
