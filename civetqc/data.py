import os

from abc import ABC, abstractmethod

import numpy as np
import pandas as pd

from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

from . import PACKAGE_DIR
from .exceptions import DuplicateIdentifierError, MissingVariableError, NegativeQCRatingError


class BaseData(ABC):
    """ inherited by all data classes """

    idvar = "ID"
    
    def __init__(self, filepath: str) -> None:
        self.df = pd.read_csv(filepath)
        self.filepath = filepath
        self.check_required_vars()
        self.check_ids_unique()
    
    def check_required_vars(self):
        missing_vars = [var for var in self.required_vars if var not in self.df.columns]
        if missing_vars != []:
            raise MissingVariableError(f"File '{self.filepath}' does not contain required columns: " + str(missing_vars).strip('[]'))
    
    def check_ids_unique(self):
        duplicate_ids = [s for s in self.df[self.idvar] if s not in self.df[self.idvar].unique()]
        if duplicate_ids != []:
            raise DuplicateIdentifierError(f"File '{self.filepath}' contains duplicate values for id variable : " + str(duplicate_ids).strip('[]'))
    
    @classmethod
    @property
    @abstractmethod
    def required_vars(self):
        pass

    def save(self, filepath):
        self.check_required_vars()
        self.check_ids_unique()
        self.df.to_csv(filepath, index=False)


class CIVETData(BaseData):
    
    feature_names = [
        "MASK_ERROR", "WM_PERCENT", "GM_PERCENT", "CSF_PERCENT", "SC_PERCENT",
        "BRAIN_VOL", "CEREBRUM_VOL", "CORTICAL_GM", "WHITE_VOL", "SUBGM_VOL",
        "SC_VOL", "CSF_VENT_VOL", "LEFT_WM_AREA", "LEFT_MID_AREA", "LEFT_GM_AREA",
        "RIGHT_WM_AREA", "RIGHT_MID_AREA", "RIGHT_GM_AREA", "GI_LEFT", "GI_RIGHT",
        "LEFT_INTER", "RIGHT_INTER", "LEFT_SURF_SURF", "RIGHT_SURF_SURF", "LAPLACIAN_MIN",
        "LAPLACIAN_MAX", "LAPLACIAN_MEAN", "GRAY_LEFT_RES", "GRAY_RIGHT_RES"
    ]
    
    def __init__(self, filepath):
        super().__init__(filepath)
    
    @property
    def features(self):
        return self.df[self.feature_names].to_numpy()
    
    @classmethod
    @property
    def required_vars(cls):
        return [cls.idvar] + cls.feature_names


class QCData(BaseData):
    """ data from QC ratings file """

    qcvar = "QC"

    def __init__(self, filepath: str) -> None:
        super().__init__(filepath)
    
    @property
    def target(self):
        return self.df[self.qcvar].to_numpy()
    
    @classmethod
    @property
    def required_vars(cls):
        return [cls.idvar, cls.qcvar]


class Dataset(CIVETData, QCData):
    
    def __init__(self, df):
        self.df = df
        X_train, X_test, y_train, y_test = train_test_split(
            self.features, self.target, test_size=0.3)
        self.train, self.test = {}, {}
        self.train["Features"], self.train["Target"] = X_train, y_train
        self.test["Features"], self.test["Target"] = X_test, y_test
    
    def get_train_test_split(self):
        return self.train["Features"], self.test["Features"], self.train["Target"], self.test["Target"]
    
    @classmethod
    def build(cls, cutoff_value: int = 1):
        studies = []
        for name in ["FEP", "INSIGHT", "LAM", "TOPSY", "NUSDAST"]:
            if cutoff_value <= 0:
                raise ValueError("Cutoff value must be greater than zero")
            path_civet_data = os.path.join(PACKAGE_DIR.parent, "data", "raw", name, f"{name}_civet_data.csv")
            path_qc_data = os.path.join(PACKAGE_DIR.parent, "data", "raw", name, f"{name}_qc_data.csv")
            for filepath in path_civet_data, path_qc_data:
                if not os.path.isfile(filepath):
                    raise FileNotFoundError(f"File not found: {filepath}. Note that this method was not designed for end user implementation!")
            civet_data = CIVETData(path_civet_data)
            qc_data = QCData(path_qc_data)
            df = pd.merge(civet_data.df, qc_data.df, on=cls.idvar).dropna()
            df = df[cls.required_vars]
            df[cls.qcvar] = df[cls.qcvar].apply(pd.to_numeric, errors='coerce')
            if not all(df[cls.qcvar] >= 0):
                raise NegativeQCRatingError(f"All QC ratings in file {path_qc_data} must be greater than zero")
            df[cls.qcvar] = np.where(df[cls.qcvar] < cutoff_value, 1, 0)
            assert all([x in range(cutoff_value + 1) for x in df[cls.qcvar]])
            studies.append(df)
        cols = [list(df.columns) for df in studies]
        assert all([l == cols[0] for l in cols[1:]])
        df = pd.concat([df for df in studies])
        df[cls.idvar] = list(range(1, len(df) + 1))
        return cls(df)
    
    @classmethod
    def make_synthetic(cls):
        X, y = make_classification(
            n_samples=1000,
            n_features=len(cls.feature_names),
            weights=(.95, .05)
        )
        df = pd.DataFrame(np.hstack([X, y.reshape((-1, 1))]), columns=cls.feature_names + [cls.qcvar])
        df[cls.idvar] = list(range(1, len(df) + 1))
        return cls(df)
    
    @classmethod
    @property
    def required_vars(cls):
        return [cls.idvar, cls.qcvar] + cls.feature_names