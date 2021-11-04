import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


class VariableNotFoundError(Exception):
    """ raised when a required variable is not found in CSV file """
    pass


class DuplicateIdentifierError(Exception):
    """ raised when a value for the ID variable appears more than once """
    pass


class DataPartition:
    """ container for test and training partitions of target or features """
    def __init__(self, train: np.ndarray, test: np.ndarray) -> None:
        assert isinstance(train, np.ndarray) and isinstance(test, np.ndarray)
        self.train, self.test = train, test


class Dataset:
    """ 
    Class used to import and organize data before modeling
    
    ...

    Attributes
    ----------

    idvar : str
        the name of the ID variable which must be in imported csv files
    qcvar : str
        the name of the column containing user QC ratings
    civet_feature_names : list
        a list of the names of the features outputted by CIVET
    required_civet_vars : list
        a list of variable names required to be in the CIVET output file
    required_user_vars : list
        a list of variable names required to be in the user ratings file
    df : pd.DataFrame
        merged dataframe created from CIVET output and user ratings
    features : DataPartition
        contains testing and training sets of features
    target : DataPartition
        contains testing and training sets of the target
    
    Operators
    ---------

    __eq__(self, other: object)
        Returns True if other is an object of self.__class__ and idvar,
        qcvar, and df are equal in both objects

    __str__(self)
        TBD
    
    Instance Methods
    ----------------
    
    all_in_range(self, var: str, r: int)
        returns whether all values in self.df[var] are in range(r)

    Static Methods
    --------------

    vars_in_cols(df: pd.DataFrame, list_vars: list, filename: str)
        raises a VariableNotFoundError if all strings in list_vars are not in df.columns
    is_unique(s: pd.Series, var_name: str, filename: str)
        raises a DuplicateIdentifierError if there are any non-unique values in s
    get_array_counts(arr: np.ndarray)
        returns a dictionary with the number of occurrences of each value in arr

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

        self.vars_in_cols(civet_data, self.required_civet_vars, civet_csv)
        self.vars_in_cols(user_ratings, self.required_user_vars, user_csv)

        self.is_unique(civet_data[self.idvar], self.idvar, civet_csv)
        self.is_unique(user_ratings[self.idvar], self.idvar, user_csv)
    
        self.df = pd.merge(civet_data, user_ratings, on=self.idvar).dropna()
        self.df[self.qcvar] = self.df[self.qcvar].apply(pd.to_numeric, errors='coerce')

        if not all(self.df[self.qcvar] >= 0):
            raise ValueError("Negative values are not permitted for QC ratings")
        
        self.df[self.qcvar] = np.where(self.df[self.qcvar] < cutoff_value, 0, 1)
        assert self.all_in_range(self.qcvar, 2)

        features_array = self.df[self.civet_feature_names].to_numpy()
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

    def all_in_range(self, var: str, r: int) -> bool:
        for value in self.df[var]:
            if value not in range(r):
                return False
        return True
    
    @staticmethod
    def vars_in_cols(df: pd.DataFrame, list_vars: list, filename: str) -> None:
        assert isinstance(df, pd.DataFrame) and isinstance(list_vars, list)
        for var in list_vars:
            if var not in df.columns:
                raise VariableNotFoundError(f"Required variable {var} not found in file {filename}")
    
    @staticmethod
    def is_unique(s: pd.Series, var_name: str, filename: str) -> None:
        assert isinstance(s, pd.Series)
        unique_values = []
        duplicated_values = []
        for value in s:
            if value not in unique_values:
                unique_values.append(value)
            else:
                duplicated_values.append(value)
        if len(s.unique()) != len(unique_values):
            raise AssertionError
        elif len(duplicated_values) != 0:
            raise DuplicateIdentifierError(f"Non-unique values {duplicated_values} for {var_name} in file {filename}")
    
    @staticmethod
    def get_array_counts(arr: np.ndarray) -> dict:
        assert isinstance(arr, np.ndarray) and arr.ndim == 1
        counts_array = np.array(np.unique(arr, return_counts=True)).T
        list_strings = []
        for i in range(len(counts_array)):
            list_strings.append(f"{counts_array[i, 0]}: {counts_array[i, 1]}")
        return "\n".join(list_strings)