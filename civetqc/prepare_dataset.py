import os
import pandas as pd
from typing import Union


class InvalidFileTypeError(Exception):
    pass

class VariableNotFoundError(Exception):
    pass


def read_csv(path_csv: str, required_vars: dict = {}) -> pd.DataFrame:
    """ wrapper for pd.check_csv that checks required data is present """

    # Ensure file exists and has .csv extension
    if path_csv.split(".")[-1] != "csv":
        raise InvalidFileTypeError(f"The file '{path_csv}' must be in csv format")
    if not os.path.isfile(path_csv):
        raise FileNotFoundError(f"The file '{path_csv}' does not exist")

    # Import csv file and check that all required variables exist
    df = pd.read_csv(path_csv)
    for key in required_vars:
        if required_vars[key] not in df.columns:
            raise VariableNotFoundError(f"Required field '{required_vars[key]}' not found in file {path_csv}")
    return df


def check_unique(s: pd.Series) -> bool:
    """ return whether all values in series are unique """
    return len(s.unique()) == len(s)


def write_csv(df: pd.DataFrame, output_dir: str, dir_name: str = "civetqc", filename: str = "df.csv") -> None:
    """ write csv file for documentation """

    if not os.path.isdir(output_dir):
        raise NotADirectoryError(f"output directory {output_dir} does not exist")

    if not os.path.isdir(os.path.join(output_dir, dir_name)):
        os.mkdir(os.path.join(output_dir, dir_name))

    df.to_csv(path_or_buf = os.path.join(output_dir, dir_name, filename))


def prepare_dataset(path_civet_output: str, path_user_ratings: str, req_usr_vars: dict, 
                    output_dir: Union[str, None] = None) -> pd.DataFrame:
    """ import civet QC output and user ratings and return merged dataframe """
    civet_output = read_csv(path_civet_output)
    user_ratings = read_csv(path_user_ratings, req_usr_vars)

    # Coerce all non-numeric values in user_ratings to NA
    user_ratings[req_usr_vars["QCVAR"]] = user_ratings[req_usr_vars["QCVAR"]].apply(pd.to_numeric, errors='coerce')

    # Make sure all ID variable values are unique
    if not check_unique(civet_output[req_usr_vars["IDVAR"]]):
        raise AssertionError("Non-unique values for ID variable in CIVET QC output")
    if not check_unique(user_ratings[req_usr_vars["IDVAR"]]):
        raise ValueError("Non-unique values for ID variable in QC ratings file")

    # Combine dataframes
    df = pd.merge(civet_output, user_ratings, on=req_usr_vars["IDVAR"])

    # Write to file if specified
    if output_dir is not None:
        write_csv(df, output_dir)

    # Combine dataframes
    return df