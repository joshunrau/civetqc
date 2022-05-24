from abc import ABC, abstractmethod

import pandas as pd

from .exceptions import DuplicateIdentifierError, MissingVariableError


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
