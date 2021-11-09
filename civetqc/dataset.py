import os
from typing import Union
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


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


class StudyData:
    """ 
    This class contains data from one CIVET output and one QC ratings file.
    
    ...

    Attributes
    ----------
    idvar : str
        Name of the ID variable which must be in imported csv files
    qcvar : str
        Name of the column containing user QC ratings
    civet_data: pd.DataFrame
        Dataframe created from CSV file outputted by CIVET
    qc_data: str
        Dataframe created from QC ratings associated with a study
    feature_names : list
        List of the names of the features outputted by CIVET
    required_civet_vars : list
        List of variable names required to be in the CIVET output file
    required_user_vars : list
        List of variable names required to be in the user ratings file
    required_all_vars : list
        List of variable names that must be in self.df
    df : pd.DataFrame
        Merged dataframe created from CIVET output and user ratings

    Methods
    ----------------
    all_in_range(self, var: str, r: int)
        Returns whether all values in self.df[var] are in range(r)
    vars_in_cols(df: pd.DataFrame, list_vars: list, filename: str)
        Raises a VariableNotFoundError if all strings in list_vars are not in df.columns

    """

    idvar = "ID"
    qcvar = "QC"
    feature_names = [
        "MASK_ERROR", "WM_PERCENT", "GM_PERCENT", "CSF_PERCENT", "SC_PERCENT",
        "BRAIN_VOL", "CEREBRUM_VOL", "CORTICAL_GM", "WHITE_VOL", "SUBGM_VOL",
        "SC_VOL", "CSF_VENT_VOL", "LEFT_WM_AREA", "LEFT_MID_AREA", "LEFT_GM_AREA",
        "RIGHT_WM_AREA", "RIGHT_MID_AREA", "RIGHT_GM_AREA", "GI_LEFT", "GI_RIGHT",
        "LEFT_INTER", "RIGHT_INTER", "LEFT_SURF_SURF", "RIGHT_SURF_SURF", "LAPLACIAN_MIN",
        "LAPLACIAN_MAX", "LAPLACIAN_MEAN", "GRAY_LEFT_RES", "GRAY_RIGHT_RES"
    ]
    required_civet_vars = [idvar] + feature_names
    required_user_vars = [idvar, qcvar]
    required_all_vars = [idvar, qcvar] + feature_names

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
        if not len(self.civet_data[self.idvar].unique()) == len(self.civet_data[self.idvar]):
            raise DuplicateIdentifierError(f"Non-unique values for ID variable in file {civet_csv}")
        if not len(self.qc_data[self.idvar].unique()) == len(self.qc_data[self.idvar]):
            raise DuplicateIdentifierError(f"Non-unique values for ID variable in file {qc_csv}")

        # ValueError may be raised if the data types are different
        try:
            self.df = pd.merge(self.civet_data, self.qc_data, on=self.idvar).dropna()
        except ValueError as err:
            raise DataFrameMergerError(f"Error merging dataframes from files '{civet_csv}' and '{qc_csv}, {err}")
        
        self.df = self.df[[self.idvar, self.qcvar] + self.feature_names]
        self.df[self.qcvar] = self.df[self.qcvar].apply(pd.to_numeric, errors='coerce')

        if not all(self.df[self.qcvar] >= 0):
            raise NegativeQCRatingError
        if cutoff_value < 1:
            raise InvalidCutoffError
        
        self.df[self.qcvar] = np.where(self.df[self.qcvar] < cutoff_value, 0, 1)
        assert self.all_in_range(self.qcvar, cutoff_value + 1)

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


class DataPartition:
    """ container for test and training sets """
    def __init__(self, features: np.ndarray, target: np.ndarray) -> None:
        assert isinstance(features, np.ndarray) and isinstance(target, np.ndarray)
        assert features.ndim == 2 and target.ndim == 1
        self.features, self.target = features, target


class Dataset(StudyData):
    """
    This class contains merged data from several studies.

    ...

    Attributes
    ----------
    studies_dir: str
        Path to directory with subdirectories for all studies
    study_filepaths: dict
        Paths to the CSV files for each study that will be imported
    df: pd.DataFrame
        Dataframe containing merged data from all studies
    features: np.ndarray
        2D array of features
    target: np.ndarray
        1D array of targets
    train: DataPartition
        Training set
    test: DataPartition
        Testing set
    """

    studies_dir = "/Users/joshua/Developer/civetqc/data/studies"

    study_filepaths = {
        "FEP": (
            os.path.join(studies_dir, "FEP", "FEP_civet_data.csv"),
            os.path.join(studies_dir, "FEP", "FEP_QC.csv")
        ),
        "LAM": (
            os.path.join(studies_dir, "LAM", "LAM_civet_data.csv"),
            os.path.join(studies_dir, "LAM", "LAM_QC.csv")
        ),
        "INSIGHT": (
            os.path.join(studies_dir, "INSIGHT", "INSIGHT_civet_data.csv"),
            os.path.join(studies_dir, "INSIGHT", "INSIGHT_QC.csv")
        ),
        "TOPSY": (
            os.path.join(studies_dir, "TOPSY", "TOPSY_civet_data.csv"),
            os.path.join(studies_dir, "TOPSY", "TOPSY_QC.csv")
        )
    }

    def __init__(self, cutoff_value: int = 1, balanced: bool = False, list_features: Union[None, list] = None)  -> None:
        """
        Parameters
        ----------
        cutoff_value: int
            the cutoff value for a valid scan
        balanced: bool
            specify whether target classes should be balanced
        list_features: None/list
            list of features to include (if None, will include all)
        """

        self.df = None
        for key in self.study_filepaths:
            if self.df is None:
                super().__init__(self.study_filepaths[key][0], self.study_filepaths[key][1], cutoff_value)
            else:
                study_data = StudyData(self.study_filepaths[key][0], self.study_filepaths[key][1])
                self.df = pd.concat([self.df, study_data.df])

        assert self.df[self.idvar].is_unique
        assert self.all_in_range(self.qcvar, cutoff_value + 1)
        self.vars_in_cols(self.df, self.required_all_vars)
        self.df[self.idvar] = list(range(1, len(self.df[self.idvar]) + 1))

        if balanced:
            min_cls = self.df[self.qcvar].value_counts().min()
            self.df = self.df.groupby(self.qcvar).sample(n=min_cls).sort_values(by=self.idvar)
        
        if list_features is not None:
            self.vars_in_cols(self.df, list_features)
            self.feature_names = list_features
            self.required_all_vars = [self.idvar, self.qcvar] + self.feature_names
            self.df = self.df[self.required_all_vars]

        self.features = self.df[self.feature_names].to_numpy()
        self.target = self.df[self.qcvar].to_numpy()

        X_train, X_test, y_train, y_test = train_test_split(self.features, self.target, random_state=0)
        self.train = DataPartition(X_train, y_train)
        self.test = DataPartition(X_test, y_test)