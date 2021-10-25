import pandas as pd
from .exceptions import *
from os.path import isfile


def import_dataset(path_csv: str, required_vars: list = []) -> pd.DataFrame:
    """ import csv file and check required data is present """

    # Ensure file exists and has .csv extension
    if path_csv.split(".")[-1] != "csv":
        raise InvalidFileTypeError(f"The file '{path_csv}' must be in csv format")
    if not isfile(path_csv):
        raise FileNotFoundError(f"The file '{path_csv}' does not exist")

    # Import csv file and check that all required variables exist
    df = pd.read_csv(path_csv)
    for var in required_vars:
        if var not in df.columns:
            raise VariableNotFoundError(f"Required field '{var}' not found in file {path_csv}")
    
    return df