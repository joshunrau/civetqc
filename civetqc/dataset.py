import os
import pandas as pd


class InvalidFileTypeError(Exception):
    pass

class VariableNotFoundError(Exception):
    pass

class DuplicateIdentifierError(Exception):
    pass


class Dataset:

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
    
    def print_data(self) -> None:
        """ print all rows and columns in pandas dataframe """
        with pd.option_context('display.max_rows', None, 'display.max_columns', None):
            print(self.data)
    
    @staticmethod
    def is_unique(s: pd.Series) -> bool:
        """ return whether all values in series are unique """
        return len(s.unique()) == len(s)
        
    @staticmethod
    def is_csv(filepath) -> bool:
        return filepath.split(".")[-1] == "csv"


class CivetOutput(Dataset):
    features = [
        "MASK_ERROR", "WM_PERCENT", "GM_PERCENT", "CSF_PERCENT", "SC_PERCENT", 
        "BRAIN_VOL", "CEREBRUM_VOL", "CORTICAL_GM", "WHITE_VOL", "SUBGM_VOL", 
        "SC_VOL", "CSF_VENT_VOL", "LEFT_WM_AREA", "LEFT_MID_AREA", "LEFT_GM_AREA", 
        "RIGHT_WM_AREA", "RIGHT_MID_AREA", "RIGHT_GM_AREA", "GI_LEFT", "GI_RIGHT", 
        "LEFT_INTER", "RIGHT_INTER", "LEFT_SURF_SURF", "RIGHT_SURF_SURF", "LAPLACIAN_MIN",
        "LAPLACIAN_MAX", "LAPLACIAN_MEAN", "GRAY_LEFT_RES", "GRAY_RIGHT_RES"
        ]
    def __init__(self, path_csv: str) -> None:
        super().__init__(path_csv, [self.idvar] + self.features)
        

class UserRatings(Dataset):
    qcvar = "QC"
    def __init__(self, path_csv: str) -> None:
        super().__init__(path_csv, [self.idvar, self.qcvar])


class MergedDataset(CivetOutput, UserRatings):
    def __init__(self, civet_output: CivetOutput, user_ratings: UserRatings) -> None:
        self.data = pd.merge(civet_output.data, user_ratings.data, on=self.idvar)

