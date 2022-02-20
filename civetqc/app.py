import argparse
import os
import pickle

from sklearn.base import BaseEstimator, is_classifier



def process_arguments(args: list) -> tuple:
    parser = argparse.ArgumentParser(prog="civetqc")
    parser.add_argument("path_csv", help="path to csv file outputted by CIVET")
    parser.add_argument(
        "output_dir", help="path to directory where results should be outputted"
    )
    parsed_args = parser.parse_args(args)

    if not os.path.isfile(parsed_args.path_csv):
        raise FileNotFoundError
    if not os.path.isdir(parsed_args.output_dir):
        raise NotADirectoryError

    return parsed_args.path_csv, parsed_args.output_dir


def load_saved_model() -> BaseEstimator:
    with open(Filepaths.saved_model, "rb") as f:
        clf = pickle.load(f)
    if not is_classifier(clf):
        raise TypeError(f"Expected sklearn classifier object, not {type(clf)} ")
    return clf
