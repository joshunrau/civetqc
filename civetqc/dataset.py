import os
import pandas as pd
import typing


class InvalidFileFormatError(Exception):
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
    
    qcvar : str
        the name of the column containing user QC ratings

    civet_vars : list
        the features outputted by CIVET

    data : pd.DataFrame
        dataframe imported from path_csv
    
    Methods
    -------

    __eq__(self, o: object)
        Returns True if o is an object of self.__class__ and idvar, 
        qcvar, and data are equal in both objects

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
    qcvar = "QC"
    
    civet_vars = [
        "MASK_ERROR", "WM_PERCENT", "GM_PERCENT", "CSF_PERCENT", "SC_PERCENT", 
        "BRAIN_VOL", "CEREBRUM_VOL", "CORTICAL_GM", "WHITE_VOL", "SUBGM_VOL", 
        "SC_VOL", "CSF_VENT_VOL", "LEFT_WM_AREA", "LEFT_MID_AREA", "LEFT_GM_AREA", 
        "RIGHT_WM_AREA", "RIGHT_MID_AREA", "RIGHT_GM_AREA", "GI_LEFT", "GI_RIGHT", 
        "LEFT_INTER", "RIGHT_INTER", "LEFT_SURF_SURF", "RIGHT_SURF_SURF", "LAPLACIAN_MIN",
        "LAPLACIAN_MAX", "LAPLACIAN_MEAN", "GRAY_LEFT_RES", "GRAY_RIGHT_RES"
        ]

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
            raise InvalidFileFormatError(f"File '{path_csv}' must be in csv format")

        self.data = pd.read_csv(path_csv)

        if not self.vars_in_cols(required_vars):
            raise VariableNotFoundError(f"Required variable not found in file {path_csv}")
        if not self.is_unique(self.data[self.idvar]):
            raise DuplicateIdentifierError(f"Non-unique values for ID variable in file {path_csv}")

    def __eq__(self, o: object) -> bool:
        if not isinstance(o, self.__class__):
            return False
        return self.idvar == o.idvar and self.qcvar == o.qcvar and self.data.equals(o.data)

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