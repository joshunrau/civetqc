from .dataset import Dataset
import numpy as np
import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import precision_score, recall_score, f1_score


class Model:

    def __init__(self) -> None:
        self.clf = None
        self.pred = None
        self.precision_score = None
        self.recall_score = None
        self.f1_score = None
    
        #precision = precision_score(self.dataset.targ_test, predicted_target, average='binary')
        #print(f"Precision: {round(precision, 2)}")

        #recall = recall_score(self.dataset.targ_test, predicted_target, average='binary')
        #print(f"Recall: {round(recall, 2)}")

        #f1 = f1_score(self.dataset.targ_test, predicted_target, average='binary')
        #print(f"F1: {round(f1, 2)}")

    


class Modeler(Dataset):

    def __init__(self, civet_csv: str, user_csv: str, cutoff_value: int = 1) -> None:
        super().__init__(civet_csv, user_csv, cutoff_value)
        self.models = {
            "LogisticRegression": None,
            "KNN": None
        }
    
    def __str__(self) -> str:
        return (
            "----------------------------------------------------------------------\n"
            "QC Value Percentages:\n"
            f"0: {round(sum(self.df['QC'] == 0) / len(self.df) * 100, 2)}%\n"
            f"1: {round(sum(self.df['QC'] == 1) / len(self.df) * 100, 2)}%\n"
            "----------------------------------------------------------------------\n"
        )

    def logistic_regression(self):
        self.regression_clf = LogisticRegression().fit(self.feat_train, self.targ_train)
        self.regression_pred = self.regression_clf.predict(self.targ_test)

    
    
    def knn(self):
        knn = KNeighborsClassifier(n_neighbors=6)
        knn.fit(self.dataset.feat_train, self.dataset.targ_train)
        predicted_target = knn.predict(self.dataset.feat_test)


    @staticmethod
    def save_model(model, filepath):
        with open(filepath, 'wb') as f:
            pickle.dump(model, f)
    @staticmethod
    def load_model(filepath):
        with open(filepath, 'rb') as f:
            model = pickle.load(f)
        return model



        
