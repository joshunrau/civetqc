import os
from typing import Union
import numpy as np
import pandas as pd


STUDIES_DIR = "/Users/joshua/Developer/civetqc/data/studies"
OUTPUT_FILE = os.path.join(STUDIES_DIR, "master_dataset.csv")


STUDY_FILEPATHS = {
    "FEP": (
        os.path.join(STUDIES_DIR, "FEP", "FEP_civet_data.csv"), 
        os.path.join(STUDIES_DIR, "FEP", "FEP_QC.csv")
        ),
    "LAM": (
        os.path.join(STUDIES_DIR, "LAM", "LAM_civet_data.csv"), 
        os.path.join(STUDIES_DIR, "LAM", "LAM_QC.csv")
        ),
    "INSIGHT": (
        os.path.join(STUDIES_DIR, "INSIGHT", "INSIGHT_civet_data.csv"), 
        os.path.join(STUDIES_DIR, "INSIGHT", "INSIGHT_QC.csv")
        ),
    "TOPSY": (
        os.path.join(STUDIES_DIR, "TOPSY", "TOPSY_civet_data.csv"), 
        os.path.join(STUDIES_DIR, "TOPSY", "TOPSY_QC.csv")
        )
}


class VariableNotFoundError(Exception):
    """ raised when a required variable is not found in CSV file """
    pass


class DuplicateIdentifierError(Exception):
    """ raised when a value for the ID variable appears more than once """
    pass


class InvalidCutoffError(ValueError):
    """ raised when cutoff value below one is provided """
    pass


class NegativeQCRatingError(ValueError):
    """ raised when negative QC value is in CSV file """
    pass


class DataFrameMergerError(ValueError):
    """ raised when cannot merge dataframes on key var due to type """
    pass


class Dataset:
    """ 
    This class contains data from one CIVET output and one QC ratings file.
    
    ...

    Attributes
    ----------
    idvar : str
        Name of the ID variable which must be in imported csv files
    qcvar : str
        Name of the column containing user QC ratings
    civet_feature_names : list
        List of the names of the features outputted by CIVET
    required_civet_vars : list
        List of variable names required to be in the CIVET output file
    required_user_vars : list
        List of variable names required to be in the user ratings file
    required_all_vars : list
        List of variable names that must be in self.df
    df : pd.DataFrame
        Merged dataframe created from CIVET output and user ratings

    Instance Methods
    ----------------
    all_in_range(self, var: str, r: int)
        Returns whether all values in self.df[var] are in range(r)
    
    Static Methods
    --------------
    vars_in_cols(df: pd.DataFrame, list_vars: list, filename: str)
        Raises a VariableNotFoundError if all strings in list_vars are not in df.columns
    
    """

    idvar = "ID"
    qcvar = "QC"
    civet_feature_names = [
        "MASK_ERROR", "WM_PERCENT", "GM_PERCENT", "CSF_PERCENT", "SC_PERCENT",
        "BRAIN_VOL", "CEREBRUM_VOL", "CORTICAL_GM", "WHITE_VOL", "SUBGM_VOL",
        "SC_VOL", "CSF_VENT_VOL", "LEFT_WM_AREA", "LEFT_MID_AREA", "LEFT_GM_AREA",
        "RIGHT_WM_AREA", "RIGHT_MID_AREA", "RIGHT_GM_AREA", "GI_LEFT", "GI_RIGHT",
        "LEFT_INTER", "RIGHT_INTER", "LEFT_SURF_SURF", "RIGHT_SURF_SURF", "LAPLACIAN_MIN",
        "LAPLACIAN_MAX", "LAPLACIAN_MEAN", "GRAY_LEFT_RES", "GRAY_RIGHT_RES"
    ]
    required_civet_vars = [idvar] + civet_feature_names
    required_user_vars = [idvar, qcvar]
    required_all_vars = [idvar, qcvar] + civet_feature_names

    def __init__(self, civet_csv: str, qc_csv: str, cutoff_value: int = 1) -> None:
        """
        Parameters
        ----------
        civet_csv: str
            path to the csv file outputted by CIVET
        qc_csv: str
            path to the csv file containing the user's QC ratings
        cutoff_value: int
            the cutoff value for a valid scan
        """

        self.civet_data = pd.read_csv(civet_csv)
        self.qc_data = pd.read_csv(qc_csv)

        self.vars_in_cols(self.civet_data, self.required_civet_vars, civet_csv)
        self.vars_in_cols(self.qc_data, self.required_user_vars, qc_csv)

        # Check that all values for ID variable are unique
        if not self.civet_data[self.idvar].unique():
            raise DuplicateIdentifierError(f"Non-unique values for ID variable in file {civet_csv}")
        if not self.qc_data[self.idvar].unique():
            raise DuplicateIdentifierError(f"Non-unique values for ID variable in file {qc_csv}")

        # ValueError may be raised if the data types are different
        try:
            self.df = pd.merge(self.civet_data, self.qc_data, on=self.idvar).dropna()
        except ValueError as err:
            raise DataFrameMergerError(f"Error merging dataframes from files '{civet_csv}' and '{qc_csv}, {err}")
        
        self.df = self.df[[self.idvar, self.qcvar] + self.civet_feature_names]
        self.df[self.qcvar] = self.df[self.qcvar].apply(pd.to_numeric, errors='coerce')

        if not all(self.df[self.qcvar] >= 0):
            raise NegativeQCRatingError
        
        if cutoff_value < 1:
            raise InvalidCutoffError
        
        self.df[self.qcvar] = np.where(self.df[self.qcvar] < cutoff_value, 0, 1)
        assert self.all_in_range(self.qcvar, 2)

    def all_in_range(self, var: str, r: int) -> bool:
        for value in self.df[var]:
            if value not in range(r):
                return False
        return True

    @staticmethod
    def vars_in_cols(df: pd.DataFrame, list_vars: list, filename: Union[str, None] = None) -> None:
        assert isinstance(df, pd.DataFrame) and isinstance(list_vars, list)
        for var in list_vars:
            if var not in df.columns:
                if filename is None:
                    raise VariableNotFoundError(f"Required variable {var} not found in columns {df.columns}")
                raise VariableNotFoundError(f"Required variable {var} not found in file {filename}")


class MasterDataset(Dataset):

    def __init__(self, filepaths: dict, cutoff_value: int) -> None:
        if not isinstance(filepaths, dict):
            raise TypeError(f"Expected argument of {dict} but received {type(filepaths)}")
        for key in filepaths:
            if not isinstance(filepaths[key], tuple):
                raise TypeError(f"Expected argument of {tuple} but received {type(filepaths[key])}")
            if len(filepaths[key]) != 2:
                raise ValueError(f"Tuples in filepaths dict must be len 2, not len {len(filepaths[key])}")
            for fpath in filepaths[key]:
                if not isinstance(fpath, str):
                    raise TypeError(f"Expected argument of {str} but received {type(fpath)}")
                if not os.path.isfile(fpath):
                    raise FileNotFoundError(f"File at path {fpath} does not exist")
            try:
                dataset_tmp = Dataset(filepaths[key][0], filepaths[key][1], cutoff_value)
                self.df = pd.concat([self.df, dataset_tmp.df])
            except AttributeError:
                super().__init__(filepaths[key][0], filepaths[key][1], cutoff_value)
        assert self.df[self.idvar].is_unique
        self.df[self.idvar] = list(range(1, len(self.df[self.idvar]) + 1))
    
    def save_df(self, output_path):
        self.df.to_csv(output_path, index=False)


def main():
    master_dataset = MasterDataset(filepaths = STUDY_FILEPATHS, cutoff_value=1)
    master_dataset.save_df(os.path.join(STUDIES_DIR, "master_dataset.csv"))
