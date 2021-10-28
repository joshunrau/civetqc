import os
import pandas as pd
from sklearn.model_selection import train_test_split
import typing


class BaseData:
    """ 
    Class with methods to setup data from csv file for analysis
    
    ...

    Attributes
    ----------

    idvar : str
        the name of the ID variable which must be in imported csv files

    data : pd.DataFrame
        dataframe imported from path_csv
    
    Methods
    -------
    vars_in_cols(self, list_vars: list)
        returns whether all strings in list_vars are in data.columns

    write_data(self, output_dir: str, filename: str)
        writes data to filename in output_dir

    col_to_numeric(self, var: str)
        converts column in data to numeric, coercing non-numeric values to NaN

    print_data(self, var: typing.Optional[str]=None)
        prints data with all rows and columns, or data[var] with all rows
    
    is_unique(s: pd.Series)
        returns whether all values in series are unique

    is_csv(filepath)
        returns whether filepath ends with .csv

    """

    idvar = "ID"

    def __init__(self, path_csv: str, required_vars: list) -> None:
        """
        Parameters
        ----------
        path_csv: str
            path to csv file to import
        required_vars: list
            list of variables required to be in csv file
        """

        if not os.path.isfile(path_csv):
            raise FileNotFoundError(f"File '{path_csv}' does not exist")
        if not self.is_csv(path_csv):
            raise ValueError(f"File '{path_csv}' must be in csv format")

        self.data = pd.read_csv(path_csv)

        if not self.vars_in_cols(required_vars):
            raise RuntimeError(f"Required field not found in file {path_csv}")
        if not self.is_unique(self.data[self.idvar]):
            raise RuntimeError(f"Non-unique values for ID variable in file {path_csv}")
    
    def vars_in_cols(self, list_vars: list) -> bool:
        for var in list_vars:
            if var not in self.data.columns:
                return False
        return True
    
    def write_data(self, output_dir: str, filename: str) -> None:
        assert os.path.isdir(output_dir)  # should check for this in case
        self.data.to_csv(path_or_buf = os.path.join(output_dir, filename))

    def col_to_numeric(self, var: str) -> None:
        self.data[var] = self.data[var].apply(pd.to_numeric, errors='coerce')
    
    def print_data(self, var: typing.Optional[str]=None) -> None:
        if var is None:
            with pd.option_context('display.max_rows', None, 'display.max_columns', None):
                print(self.data)
        else:
            with pd.option_context('display.max_rows', None, 'display.max_columns', None):
                print(self.data[var])

    @staticmethod
    def is_unique(s: pd.Series) -> bool:
        return len(s.unique()) == len(s)
    
    @staticmethod
    def is_csv(filepath: str) -> bool:
        return filepath.split(".")[-1] == "csv"


class CivetData(BaseData):
    """ 
    Subclass of BaseData used to import CIVET QC output file
    
    ...

    Attributes
    ----------

    civet_vars : list
        the features outputted by CIVET

    """
    civet_vars = [
        "MASK_ERROR", "WM_PERCENT", "GM_PERCENT", "CSF_PERCENT", "SC_PERCENT", 
        "BRAIN_VOL", "CEREBRUM_VOL", "CORTICAL_GM", "WHITE_VOL", "SUBGM_VOL", 
        "SC_VOL", "CSF_VENT_VOL", "LEFT_WM_AREA", "LEFT_MID_AREA", "LEFT_GM_AREA", 
        "RIGHT_WM_AREA", "RIGHT_MID_AREA", "RIGHT_GM_AREA", "GI_LEFT", "GI_RIGHT", 
        "LEFT_INTER", "RIGHT_INTER", "LEFT_SURF_SURF", "RIGHT_SURF_SURF", "LAPLACIAN_MIN",
        "LAPLACIAN_MAX", "LAPLACIAN_MEAN", "GRAY_LEFT_RES", "GRAY_RIGHT_RES"
        ]
    def __init__(self, path_csv: str) -> None:
        super().__init__(path_csv, [self.idvar] + self.civet_vars)
        

class UserData(BaseData):
    """ 
    Subclass of BaseData used to import user QC ratings
    
    ...

    Attributes
    ----------

    qcvar : str
        the name of the column containing user QC ratings

    """
    qcvar = "QC"
    def __init__(self, path_csv: str) -> None:
        super().__init__(path_csv, [self.idvar, self.qcvar])


class Dataset(CivetData, UserData):
    """ 
    Class used to merge and organize data for analysis
    
    ...

    Attributes
    ----------

    data : pd.DataFrame
        merged dataframe created from CivetData and UserData objects when instantiated
    
    features : pd.DataFrame
        subset of data containing features from the CIVET QC output
    
    target: pd.Series
        subset of data containing the known QC ratings by the user

    feat_train
        training set of features (75%)
    
    feat_test
        testing set of features (25%)
    
    targ_train
        training set of target (75%)

    targ_test
        testing set of target (25%)

    """

    def __init__(self, civet_csv: str, user_csv: str) -> None:
        """
        Parameters
        ----------
        civet_csv: str
            path to the csv file outputted by CIVET
        user_csv: str
            path to the csv file containing the user's QC ratings
        """
        civet_output = CivetData(civet_csv)
        user_ratings = UserData(user_csv)
        self.data = pd.merge(civet_output.data, user_ratings.data, on=self.idvar).dropna()
        self.features = self.data[self.civet_vars]
        self.target = self.data[self.qcvar]
        self.feat_train, self.feat_test, self.targ_train, self.targ_test = train_test_split(
            self.features, self.target, random_state=0)