import os
import numpy as np
import pandas as pd
import typing
from sklearn.model_selection import train_test_split


class VariableNotFoundError(Exception):
    pass


class DuplicateIdentifierError(Exception):
    pass


class DataPartition:
    def __init__(self, train: np.ndarray, test: np.ndarray) -> None:
        assert isinstance(train, np.ndarray) and isinstance(test, np.ndarray)
        self.train, self.test = train, test


class Dataset:
    """ 
    Class with methods to setup data from csv files for analysis
    
    ...

    Attributes
    ----------

    idvar : str
        the name of the ID variable which must be in imported csv files
    
    qcvar : str
        the name of the column containing user QC ratings

    civet_vars : list
        the features outputted by CIVET

    df : pd.DataFrame
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
    
    Methods
    -------

    __eq__(self, other: object)
        Returns True if other is an object of self.__class__ and idvar,
        qcvar, and df are equal in both objects

    write_data(self, output_dir: str, filename: str)
        writes data to filename in output_dir

    col_to_numeric(self, var: str)
        converts column in data to numeric, coercing non-numeric values to NaN

    print_data(self, var: typing.Optional[str]=None)
        prints data with all rows and columns, or data[var] with all rows
    
    all_in_range(self, var: str, r: int)
        returns whether all values in self.df[var] are in range(r)

    vars_in_cols(df: pd.DataFrame, list_vars: list)
        returns whether all strings in list_vars are in df.columns
    
    is_unique(s: pd.Series)
        returns whether all values in series are unique

    """
    
    idvar = "ID"
    qcvar = "QC"

    civet_vars = [
        "MASK_ERROR", "WM_PERCENT", "GM_PERCENT", "CSF_PERCENT", "SC_PERCENT",
        "BRAIN_VOL", "CEREBRUM_VOL", "CORTICAL_GM", "WHITE_VOL", "SUBGM_VOL",
        "SC_VOL", "CSF_VENT_VOL", "LEFT_WM_AREA", "LEFT_MID_AREA", "LEFT_GM_AREA",
        "RIGHT_WM_AREA", "RIGHT_MID_AREA", "RIGHT_GM_AREA", "GI_LEFT", "GI_RIGHT",
        "LEFT_INTER", "RIGHT_INTER", "LEFT_SURF_SURF", "RIGHT_SURF_SURF", "LAPLACIAN_MIN",
        "LAPLACIAN_MAX", "LAPLACIAN_MEAN", "GRAY_LEFT_RES", "GRAY_RIGHT_RES"
    ]

    def __init__(self, civet_csv: str, user_csv: str, cutoff_value: int = 1) -> None:
        """
        Parameters
        ----------
        civet_csv: str
            path to the csv file outputted by CIVET
        user_csv: str
            path to the csv file containing the user's QC ratings
        cutoff_value: int
            the cutoff value for a valid scan
        """

        civet_data = pd.read_csv(civet_csv)
        user_ratings = pd.read_csv(user_csv)

        if not self.vars_in_cols(civet_data, [self.idvar] + self.civet_vars):
            raise VariableNotFoundError(f"Required variable not found in file {civet_csv}")
        if not self.vars_in_cols(user_ratings, [self.idvar, self.qcvar]):
            raise VariableNotFoundError(f"Required variable not found in file {user_csv}")

        if not self.is_unique(civet_data[self.idvar]):
            raise DuplicateIdentifierError(f"Non-unique values for ID variable in file {civet_csv}")
        if not self.is_unique(user_ratings[self.idvar]):
            raise DuplicateIdentifierError(f"Non-unique values for ID variable in file {user_csv}")
        
        self.df = pd.merge(civet_data, user_ratings, on=self.idvar).dropna()
        self.col_to_numeric(self.qcvar)

        if not all(self.df[self.qcvar] >= 0):
            raise ValueError("Negative values are not permitted for QC ratings")
        
        self.df[self.qcvar] = np.where(self.df[self.qcvar] == 0, 0, 1)
        assert self.all_in_range(self.qcvar, 2)

        features_array = self.df[self.civet_vars].to_numpy()
        target_array = self.df[self.qcvar].to_numpy()
        x_train, x_test, y_train, y_test = train_test_split(features_array, target_array, random_state=0)
        self.features = DataPartition(x_train, x_test)
        self.target = DataPartition(y_train, y_test)
    
    def __eq__(self, other: object) -> bool:
        if not isinstance(other, self.__class__):
            return False
        return self.idvar == other.idvar and self.qcvar == other.qcvar and self.df.equals(other.df)

    def __str__(self) -> str:
        return (
            "Dataset Object\n"
            "----------------------------------------------------------------------\n"
            "General Information:\n"
            f"Number of Observations: {len(self.df)}\n\n"
            "Target:\n"
            f"Train (N={len(self.target.train)})\n"
            f"{self.get_array_counts(self.target.train)}\n"
            f"Test\n"
            f"{self.get_array_counts(self.target.test)}\n"
        )

    def write_data(self, output_dir: str, filename: str) -> None:
        assert os.path.isdir(output_dir)  # should check for this in case
        self.df.to_csv(path_or_buf=os.path.join(output_dir, filename))

    def col_to_numeric(self, var: str) -> None:
        self.df[var] = self.df[var].apply(pd.to_numeric, errors='coerce')

    def print_data(self, var: typing.Optional[str] = None) -> None:
        if var is None:
            with pd.option_context('display.max_rows', None, 'display.max_columns', None):
                print(self.df)
        else:
            with pd.option_context('display.max_rows', None, 'display.max_columns', None):
                print(self.df[var])
    
    def all_in_range(self, var: str, r: int) -> bool:
        for value in self.df[var]:
            if value not in range(r):
                return False
        return True

    @staticmethod
    def vars_in_cols(df: pd.DataFrame, list_vars: list) -> bool:
        for var in list_vars:
            if var not in df.columns:
                return False
        return True

    @staticmethod
    def is_unique(s: pd.Series) -> bool:
        return len(s.unique()) == len(s)
    
    @staticmethod
    def get_array_counts(arr: np.ndarray) -> dict:
        assert isinstance(arr, np.ndarray) and arr.ndim == 1
        counts_array = np.array(np.unique(arr, return_counts=True)).T
        list_strings = []
        for i in range(len(counts_array)):
            list_strings.append(f"{counts_array[i, 0]}: {counts_array[i, 1]}")
        return "\n".join(list_strings)
