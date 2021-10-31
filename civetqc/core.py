from .dataset import Dataset
import numpy as np
import pickle
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier


class Modeler(Dataset):
    def __init__(self, civet_csv: str, user_csv: str, k: int):
        super().__init__(civet_csv, user_csv)

        # KNN
        self.knn = KNeighborsClassifier(n_neighbors=k)
        self.knn.fit(self.feat_train, self.targ_train)
        targ_pred = self.knn.predict(self.feat_test)
        print_target_score(targ_pred, self.targ_test)

        # Neural Network
        clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1)
        clf.fit(self.feat_train, self.targ_train)
        targ_pred = clf.predict(self.feat_test)
        print_target_score(targ_pred, self.targ_test)


def print_target_score(predicted, actual):
    print("Test set score: {:.2f}".format(np.mean(predicted == actual)))


def save_model(model, filepath):
    with open(filepath, 'wb') as f:
        pickle.dump(model, f)


def load_model(filepath):
    with open(filepath, 'rb') as f:
        model = pickle.load(f)
    return model
