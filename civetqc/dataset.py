import os
from typing import Union
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import warnings

"""

TO DO:
    Manage floating point values for QC ratings
    iRELATE: still waiting on whether QC was done
    NUSDAST: extract CIVET outputs from /data/lepage/NUSDAST/data/processed/civet2.1.0_bpipe_Niagara

    tar -xzvf /data/lepage/NUSDAST/data/processed/civet2.1.0_bpipe_Niagara/NUSDASTcivet.tar.gz /home/cic/unrjos

    
"""

STUDIES_DIR = "/Users/joshua/Developer/civetqc/data/studies"


FILEPATHS = {
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
        ),
    "NUSDAST": (
        None,
        os.path.join(STUDIES_DIR, "NUSDAST", "NUSDAST_QC.csv")
    )
}


class VariableNotFoundError(Exception):
    """ raised when a required variable is not found in CSV file """
    pass


class DuplicateIdentifierError(Exception):
    """ raised when a value for the ID variable appears more than once """
    pass


class NegativeQCRatingError(ValueError):
    """ raised when negative QC value is in CSV file """
    pass


class DataPartition:
    """ container for test and training partitions of target or features """
    def __init__(self, train: np.ndarray, test: np.ndarray) -> None:
        assert isinstance(train, np.ndarray) and isinstance(test, np.ndarray)
        self.train, self.test = train, test


class Dataset:
    """ 
    This class contains the essential attributes and methods used to import and organize 
    the data for the model. In its basic form, the Dataset class includes data from one 
    CIVET output and one QC ratings file. The class method 'master_dataset' can be used
    to create an object of type Dataset containing data from multiple studies.
    
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
    features : DataPartition
        Contains testing and training sets of features
    target : DataPartition
        Contains testing and training sets of the target

    Operators
    ---------
    __eq__(self, other: object)
        Returns True if other is an object of self.__class__ and idvar,
        qcvar, and df are equal in both objects
    __str__(self)
        Returns the number of total observations and the target counts
        for both the testing and training sets

    Instance Methods
    ----------------
    all_in_range(self, var: str, r: int)
        Returns whether all values in self.df[var] are in range(r)

    Static Methods
    --------------
    format_class_counts(d: dict)
        Returns a formatted string of the class counts in the test and training sets
    vars_in_cols(df: pd.DataFrame, list_vars: list, filename: str)
        Raises a VariableNotFoundError if all strings in list_vars are not in df.columns
    is_unique(s: pd.Series, var_name: str, filename: str)
        Raises a DuplicateIdentifierError if there are any non-unique values in s
    get_array_counts(arr: np.ndarray)
        Returns a dict with the number of occurrences of each value in arr
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
        self.df = self.df[[self.idvar, self.qcvar] + self.civet_feature_names]
        self.df[self.qcvar] = self.df[self.qcvar].apply(pd.to_numeric, errors='coerce')

        if not all(self.df[self.qcvar] >= 0):
            raise NegativeQCRatingError
        
        self.df[self.qcvar] = np.where(self.df[self.qcvar] < cutoff_value, 0, 1)
        assert self.all_in_range(self.qcvar, 2)

        features_array = self.df[self.civet_feature_names].to_numpy()
        target_array = self.df[self.qcvar].to_numpy()
        x_train, x_test, y_train, y_test = train_test_split(features_array, target_array, random_state=1)
        self.features = DataPartition(x_train, x_test)
        self.target = DataPartition(y_train, y_test)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, self.__class__):
            return False
        return self.idvar == other.idvar and self.qcvar == other.qcvar and self.df.equals(other.df)
    
    def __str__(self) -> str:
        dataset_header = "DATASET"
        horizontal_line = "----------------------------------------------------------------------"
        observations = f"Number of Observations: {len(self.df)}"
        target_train = f"Target Train:\n{self.format_class_counts(self.get_array_counts(self.target.train))}"
        target_test = f"Target Test:\n{self.format_class_counts(self.get_array_counts(self.target.test))}"
        return "\n".join([dataset_header, horizontal_line, observations, target_train, target_test])

    def all_in_range(self, var: str, r: int) -> bool:
        for value in self.df[var]:
            if value not in range(r):
                return False
        return True
    
    @staticmethod
    def format_class_counts(d: dict) -> str:
        list_strings = []
        sum_classes = 0
        for key in d:
            sum_classes += d[key]
        for key in d:
            list_strings.append(f"{key}: {d[key]} ({round(d[key]/sum_classes*100, 1)}%)")
        return "\n".join(list_strings)

    @staticmethod
    def vars_in_cols(df: pd.DataFrame, list_vars: list, filename: Union[str, None] = None) -> None:
        assert isinstance(df, pd.DataFrame) and isinstance(list_vars, list)
        for var in list_vars:
            if var not in df.columns:
                if filename is None:
                    raise VariableNotFoundError(f"Required variable {var} not found in columns {df.columns}")
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
        counts = {}
        for i in range(len(counts_array)):
            counts[counts_array[i, 0]] = counts_array[i, 1]
        return counts


class MasterDataset(Dataset):

    def __init__(self, filepaths: dict = FILEPATHS, cutoff_value: int = 1) -> None:
        
        if not isinstance(filepaths, dict):
            raise TypeError(f"Expected argument of {dict} but received {type(filepaths)}")
        for key in filepaths:
            if not isinstance(filepaths[key], tuple):
                raise TypeError(f"Expected argument of {tuple} but received {type(filepaths[key])}")
            if len(filepaths[key]) != 2:
                raise ValueError(f"Tuples in filepaths dict must be len 2, not len {len(filepaths[key])}")
            if any([x for x in filepaths[key] if x is None]):
                warnings.warn(f"Missing filepath for {key}, skipping study...")
                continue
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
        
        features_array = self.df[self.civet_feature_names].to_numpy()
        target_array = self.df[self.qcvar].to_numpy()
        x_train, x_test, y_train, y_test = train_test_split(features_array, target_array, random_state=1)
        self.features = DataPartition(x_train, x_test)
        self.target = DataPartition(y_train, y_test)


def dict_values_equal(d: dict):
    """ returns whether all values in dict are of equal length """
    req_len = len(list(d.values())[0])
    for key in d:
        if len(d[key]) != req_len:
            return False
    return True


def txt_to_csv(dir_name: str, output_dir: Union[None, str] = None) -> None:
    """ given a directory containing civet txt outputs, creates a single csv file """
    
    # Get list of patient files in directory
    patient_files = {}
    for filename in os.listdir(dir_name):
        if "civet_qc" in filename:
            patient_files[filename.split("_")[0]] = filename

    # Open each file and append data to dictionary
    civet_dict = {"ID": []}
    for patient_id in patient_files:
        civet_dict["ID"].append(patient_id)
        with open(os.path.join(dir_name, patient_files[patient_id]), 'r') as f:
            for line in f:
                var, value = line.split("=")
                try:
                    civet_dict[var].append(value.strip("\n"))
                except KeyError:
                    civet_dict[var] = [value.strip("\n")]

    # Verify length of all values are equal and write to csv
    if not dict_values_equal(civet_dict):
        raise ValueError("all values in dictionary are not equal!")
    pd.DataFrame(civet_dict).to_csv(output_dir, index=False)