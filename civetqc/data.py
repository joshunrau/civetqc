import os

from abc import ABC, abstractmethod
from typing import Union

import numpy as np
import pandas as pd


class BaseData(ABC):

    id_var = "ID"
    
    @classmethod
    @property
    @abstractmethod
    def required_vars(cls) -> list:
        pass
    
    def __init__(self, df: Union[None, pd.DataFrame] = None) -> None: 
        if df is None:
            self.df = pd.DataFrame(columns=self.required_vars)
        elif isinstance(df, pd.DataFrame):
            missing_vars = [var for var in self.required_vars if var not in df.columns]
            if missing_vars != []:
                raise ValueError(f"DataFrame does not contain required columns: " + ', '.join(missing_vars))
            self.df = df.loc[:, self.required_vars]
        else:
            raise TypeError()
        
        cols_with_missing_values = [col for col in self.df.columns if self.df[col].isnull().any()]
        if cols_with_missing_values != []:
            raise ValueError("Unexpected missing value in columns: " + ', '.join(cols_with_missing_values))
        
        duplicate_ids = [s for s in self.df[self.id_var] if s not in self.df[self.id_var].unique()]
        if duplicate_ids != []:
            raise ValueError(f"Data contains duplicate values for id variable : " + ', '.join(duplicate_ids))

        self.set_column_dtype(self.id_var, str)
    
    def append(self, other):
        if type(self) != type(other):
            raise TypeError(f"Object to append must be instance of '{self.__class__.__name__}', not '{other.__class__.__name__}'")
        assert self.df.columns.equals(other.df.columns)
        self.df = pd.concat([self.df, other.df])
    
    def set_column_dtype(self, colname, dtype):
        self.df[colname] = self.df[colname].astype(dtype)
    
    def to_csv(self, filepath, **kwargs):
        self.df.to_csv(filepath, **kwargs)
    
    @classmethod
    def from_csv(cls, filepath, **kwargs):
        return cls(pd.read_csv(filepath, **kwargs))


class CIVETData(BaseData):
    
    feature_names = [
        "MASK_ERROR", "WM_PERCENT", "GM_PERCENT", "CSF_PERCENT", "SC_PERCENT",
        "BRAIN_VOL", "CEREBRUM_VOL", "CORTICAL_GM", "WHITE_VOL", "SUBGM_VOL",
        "SC_VOL", "CSF_VENT_VOL", "LEFT_WM_AREA", "LEFT_MID_AREA", "LEFT_GM_AREA",
        "RIGHT_WM_AREA", "RIGHT_MID_AREA", "RIGHT_GM_AREA", "GI_LEFT", "GI_RIGHT",
        "LEFT_INTER", "RIGHT_INTER", "LEFT_SURF_SURF", "RIGHT_SURF_SURF", "LAPLACIAN_MIN",
        "LAPLACIAN_MAX", "LAPLACIAN_MEAN", "GRAY_LEFT_RES", "GRAY_RIGHT_RES"
    ]

    @classmethod
    @property
    def required_vars(cls):
        return [cls.id_var] + cls.feature_names
    
    @property
    def features(self):
        return self.df[self.feature_names].to_numpy()
    
    @classmethod
    def from_output_files(cls, dir_path: str, prefix: str = '', subset_subject_ids: Union[list, None] = None):

        target_file_suffix = 'civet_qc.txt'

        filepaths = {}
        for filename in os.listdir(dir_path):
            if filename.endswith(target_file_suffix):
                subject_id = filename.removeprefix(prefix).removesuffix(target_file_suffix).strip('_')
                if subset_subject_ids is None or subject_id in subset_subject_ids:
                    filepaths[subject_id] = os.path.join(dir_path, filename)
            
        data = {}
        for subject_id, filepath in filepaths.items():
            with open(filepath, 'r') as f:
                content = f.read().strip().split('\n')
            subject_data = {}
            for key, value in [line.split('=') for line in content]:
                try:
                    subject_data[key] = float(value)
                except ValueError as err:
                    raise RuntimeError(f"Unexpected non-numeric value '{value}' for variable '{key}' in file: {filepath}") from err
            missing_vars = [var for var in cls.feature_names if var not in subject_data.keys()]
            if missing_vars != []:
                raise RuntimeError(f"Missing variables in file '{filepath}': {', '.join(missing_vars)}")
            data[subject_id] = subject_data
        return cls(pd.DataFrame.from_dict(data, orient='index').rename_axis(cls.id_var).reset_index())


class QCRatingsData(BaseData):

    qc_ratings_var = "QC"

    def __init__(self, df: Union[None, pd.DataFrame] = None) -> None:
        super().__init__(df)
        self.df[self.qc_ratings_var] = self.df[self.qc_ratings_var].apply(pd.to_numeric, errors='coerce').dropna()

    def apply_cutoff(self, value: Union[int, float] = 1):
        self.df[self.qc_ratings_var] = np.where(self.df[self.qc_ratings_var] < value, 1, 0)
    
    @classmethod
    @property
    def required_vars(cls):
        return [cls.id_var, cls.qc_ratings_var]


class Dataset(CIVETData, QCRatingsData):
    
    @classmethod
    @property
    def required_vars(cls):
        return [cls.id_var, cls.qc_ratings_var] + cls.feature_names
    
    @property
    def target(self):
        return self.df[self.qc_ratings_var].to_numpy()

    @classmethod
    def from_merge(cls, civet_data: CIVETData, qc_ratings_data: QCRatingsData):
        return cls(pd.merge(civet_data.df, qc_ratings_data.df, on=cls.id_var))