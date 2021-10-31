from .dataset import Dataset
import numpy as np
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier


class Modeler(Dataset):
    """ 
    Class used to merge and organize data for analysis
    
    ...

    Attributes
    ----------

    data : pd.DataFrame
        merged dataframe created from CIVET and user data
    
    features : pd.DataFrame
        subset of data containing features from the CIVET QC output
    
    target: pd.Series
        subset of data containing the known QC ratings by the user

    feat_train: pd.DataFrame
        training set of features (75%)
    
    feat_test: pd.DataFrame
        testing set of features (25%)
    
    targ_train: pd.Series
        training set of target (75%)

    targ_test: pd. Series
        testing set of target (25%)

    """


    saved_models = {
        "KNN" : "./data/models/knn.pickle",
    }


    def __init__(self, civet_csv: str, user_csv: str) -> None:
        """
        Parameters
        ----------
        civet_csv: str
            path to the csv file outputted by CIVET
        user_csv: str
            path to the csv file containing the user's QC ratings
        """
        civet_output = Dataset(civet_csv, [self.idvar] + self.civet_vars)
        user_ratings = Dataset(user_csv, [self.idvar, self.qcvar])
        self.data = pd.merge(civet_output.data, user_ratings.data, on=self.idvar).dropna()
        self.features = self.data[self.civet_vars]
        self.target = self.data[self.qcvar]
        self.feat_train, self.feat_test, self.targ_train, self.targ_test = train_test_split(
            self.features, self.target, random_state=0)
    
    def train_knn(self, k):
        self.knn = KNeighborsClassifier(n_neighbors=k)
        self.knn.fit(self.feat_train, self.targ_train)
        targ_pred = self.knn.predict(self.feat_test)
        self.print_targ_score(targ_pred, self.targ_test)
    
    def train_nn(self):
        clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1)
        clf.fit(self.feat_train, self.targ_train)
        targ_pred = clf.predict(self.feat_test)
        self.print_targ_score(targ_pred, self.targ_test)

    @staticmethod
    def print_targ_score(predicted, actual):
        print("Test set score: {:.2f}".format(np.mean(predicted == actual)))

    @staticmethod
    def save_model(model, filepath):
        with open(filepath, 'wb') as f:
            pickle.dump(model, f)
            
    @staticmethod
    def load_model(filepath):
        with open(filepath, 'rb') as f:
            model = pickle.load(f)
        return model

