import numpy as np
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import typing


class InvalidFileTypeError(Exception):
    pass

class VariableNotFoundError(Exception):
    pass

class DuplicateIdentifierError(Exception):
    pass


class Dataset:
    """ 
    Class with methods to setup data from csv file for analysis
    
    ...

    Attributes
    ----------

    idvar : str
        the name of the ID variable which must be in imported csv files

    data : pd.DataFrame
        dataframe imported from path_csv
    
    """

    idvar = "ID"

    def __init__(self, path_csv: str, required_vars: list) -> None:
        
        # Ensure file exists and has .csv extension
        if not os.path.isfile(path_csv):
            raise FileNotFoundError(f"File '{path_csv}' does not exist")
        if not self.is_csv(path_csv):
            raise InvalidFileTypeError(f"File '{path_csv}' must be in csv format")
        
        # Read csv file and ensure all required variables are present
        self.data = pd.read_csv(path_csv)
        if not self.vars_in_cols(required_vars):
            raise VariableNotFoundError(f"Required field not found in file {path_csv}")
        
        # Ensure all values for ID variable are unique
        if not self.is_unique(self.data[self.idvar]):
            raise DuplicateIdentifierError(f"Non-unique values for ID variable in file {path_csv}")
        
    def vars_in_cols(self, list_vars) -> bool:
        for var in list_vars:
            if var not in self.data.columns:
                return False
        return True
    
    def write_data(self, output_dir: str, dir_name: str = "civetqc", filename: str = "df.csv") -> None:
        """ write csv file for documentation """

        if not os.path.isdir(output_dir):
            raise NotADirectoryError(f"output directory {output_dir} does not exist")

        if not os.path.isdir(os.path.join(output_dir, dir_name)):
            os.mkdir(os.path.join(output_dir, dir_name))

        self.data.to_csv(path_or_buf = os.path.join(output_dir, dir_name, filename))
    
    def col_to_numeric(self, var: str) -> None:
        self.data[var] = self.data[var].apply(pd.to_numeric, errors='coerce')
    
    def print_data(self, var: typing.Optional[str]=None) -> None:
        """ print all rows and columns in pandas dataframe, or all rows in column """
        if var is None:
            with pd.option_context('display.max_rows', None, 'display.max_columns', None):
                print(self.data)
        else:
            with pd.option_context('display.max_rows', None, 'display.max_columns', None):
                print(self.data[var])

    @staticmethod
    def is_unique(s: pd.Series) -> bool:
        """ return whether all values in series are unique """
        return len(s.unique()) == len(s)
        
    @staticmethod
    def is_csv(filepath) -> bool:
        return filepath.split(".")[-1] == "csv"


class CivetOutput(Dataset):
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
        

class UserRatings(Dataset):
    qcvar = "QC"
    def __init__(self, path_csv: str) -> None:
        super().__init__(path_csv, [self.idvar, self.qcvar])


class CivetData(CivetOutput, UserRatings):
    def __init__(self, civet_output: CivetOutput, user_ratings: UserRatings, drop_na: bool = False) -> None:
        self.data = pd.merge(civet_output.data, user_ratings.data, on=self.idvar)
        if drop_na:
            self.data = self.data.dropna() 
        self.features = self.data[self.civet_vars]
        self.target = self.data[self.qcvar]
        self.feat_train, self.feat_test, self.targ_train, self.targ_test = train_test_split(
            self.features, self.target, random_state=0)
    
    def test_knn(self, r: typing.Union[int, range]) -> None:
        if type(r) == int:
            r = range(r, r+1)
        for i in r:
            knn = KNeighborsClassifier(n_neighbors=i)
            knn.fit(self.feat_train, self.targ_train)
            targ_pred = knn.predict(self.feat_test)
            print(sum(self.data["QC"] == '2')/len(self.data))
            print(sum(self.data["QC"] == '1')/len(self.data))
            print(sum(self.data["QC"] == '0')/len(self.data))
            print(targ_pred)
            print(list(self.targ_test))
            print("Test set score: {:.2f}".format(np.mean(targ_pred == self.targ_test)))
