import argparse
import os
import pickle

import pkg_resources

from .data.create_dataset import CIVETData
from .exceptions import ModelNotFoundError, InvalidClassifierError


class UserArguments:

    def __init__(self, args: list) -> None:
        parser = argparse.ArgumentParser(prog="civetqc")
        parser.add_argument("path_csv", help="path to csv file outputted by CIVET")
        parser.add_argument("output_dir", help="path to directory where results should be outputted")
        self.args = parser.parse_args(args)
        self.path_csv, self.output_dir = self.args.path_csv, self.args.output_dir

    def __str__(self) -> str:
        return str(self.args)

    def verify_paths(self):
        if not os.path.isfile(self.path_csv):
            raise FileNotFoundError
        if not os.path.isdir(self.output_dir):
            raise NotADirectoryError


class UserData(CIVETData):

    qcvar = "PREDICTED_QC"
    output_filename = "civetqc.csv"
    def __init__(self, path_csv: str) -> None:
        super().__init__(path_csv)
        self.feature_names = self.feature_names
        self.features = self.df[self.feature_names].to_numpy()

    def predict_qc(self, clf, update_df: bool = False):

        try:
            self.predicted_qc = clf.predict(self.features)
        except AttributeError as err:
            raise InvalidClassifierError(f"Expected sklearn classifier object for 'clf', not {type(clf)}") from err

        if update_df:
            self.df[self.qcvar] = self.predicted_qc
        else:
            print(self.predicted_qc)

    def write_csv(self, dir_name: str):
        if self.qcvar not in self.df.columns:
            raise ValueError(f"UserData object requires {self.qcvar} in dataframe before writing to file")
        self.df.to_csv(os.path.join(dir_name, self.output_filename), index=False)


class SavedModel:

    path = pkg_resources.resource_filename(__name__, 'model/model.pkl')

    def __init__(self) -> None:
        try:
            with open(self.path, 'rb') as f:
                self.clf = pickle.load(f)
        except FileNotFoundError as err:
            raise ModelNotFoundError from err


def main(args: list):

    # Process user arguments
    usr_args = UserArguments(args)
    usr_args.verify_paths()

    # Load saved model
    clf = SavedModel().clf

    # Use model to predict
    data = UserData(usr_args.path_csv)
    data.predict_qc(clf, update_df=True)
    data.write_csv(usr_args.output_dir)