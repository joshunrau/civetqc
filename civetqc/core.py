from .dataset import Dataset
import numpy as np
import pickle
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import precision_score


class SaveData:
    def save_model(model, filepath):
        with open(filepath, 'wb') as f:
            pickle.dump(model, f)
    def load_model(filepath):
        with open(filepath, 'rb') as f:
            model = pickle.load(f)
        return model


class Modeler(Dataset):
    def __init__(self, civet_csv: str, user_csv: str, k: int):

        super().__init__(civet_csv, user_csv)

        # KNN
        self.knn = KNeighborsClassifier(n_neighbors=k)
        self.knn.fit(self.feat_train, self.targ_train)
        self.targ_pred = self.knn.predict(self.feat_test)

        # calculate precision for binary classification problem
        precision = precision_score(self.targ_test, self.targ_pred, average='binary')
        print(precision)

        print_target_score(self.targ_pred, self.targ_test)


def print_target_score(predicted, actual):
    print("Test set score: {:.2f}".format(np.mean(predicted == actual)))


        
