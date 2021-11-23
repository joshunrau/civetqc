import os
from abc import ABC, abstractmethod

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, is_classifier


class VariableNotFoundError(Exception):
    """ raised when a required variable is not found in CSV file """
    pass


class DuplicateIdentifierError(ValueError):
    """ raised when a value for the ID variable appears more than once """
    pass


class DataFrameMergerError(Exception):
    """ raised when cannot merge dataframes on key var due to type """
    pass


class NegativeQCRatingError(ValueError):
    """ raised when negative QC value is in CSV file """
    pass


class BaseData(ABC):

    idvar = "ID"

    def __init__(self, path_csv: str) -> None:
        self.path_csv = path_csv
        self.df = pd.read_csv(path_csv)
        self.check_required_vars()
        self.check_ids_unique()

    def check_required_vars(self):
        for var in self.required_vars:
            if var not in self.df.columns:
                try:
                    raise VariableNotFoundError(f"Required variable {var} not found in file {self.path_csv}")
                except AttributeError as err:
                    raise AssertionError from err

    def check_ids_unique(self):
        if not len(self.df[self.idvar].unique()) == len(self.df[self.idvar]):
            try:
                raise DuplicateIdentifierError(f"Non-unique values for ID variable in file {self.path_csv}")
            except AttributeError as err:
                raise AssertionError from err

    @property
    @abstractmethod
    def required_vars(self):
        pass


class CIVETData(BaseData):
    """ data from CIVET output file """

    feature_names = [
        "MASK_ERROR", "WM_PERCENT", "GM_PERCENT", "CSF_PERCENT", "SC_PERCENT",
        "BRAIN_VOL", "CEREBRUM_VOL", "CORTICAL_GM", "WHITE_VOL", "SUBGM_VOL",
        "SC_VOL", "CSF_VENT_VOL", "LEFT_WM_AREA", "LEFT_MID_AREA", "LEFT_GM_AREA",
        "RIGHT_WM_AREA", "RIGHT_MID_AREA", "RIGHT_GM_AREA", "GI_LEFT", "GI_RIGHT",
        "LEFT_INTER", "RIGHT_INTER", "LEFT_SURF_SURF", "RIGHT_SURF_SURF", "LAPLACIAN_MIN",
        "LAPLACIAN_MAX", "LAPLACIAN_MEAN", "GRAY_LEFT_RES", "GRAY_RIGHT_RES"
    ]

    def __init__(self, path_csv: str) -> None:
        super().__init__(path_csv)
        self.features = self.df[self.feature_names].to_numpy()

    def predict_qc(self, clf: BaseEstimator):
        if not is_classifier(clf):
            raise TypeError(f"Expected sklearn classifier object, not {type(clf)}")
        self.df["QC"] = clf.predict(self.features)

    def write_csv(self, output_dir):
        filepath = os.path.join(output_dir, "civetqc.csv")
        self.df.to_csv(filepath, index=False)
        print(f"Output file: {filepath}")

    @property
    def required_vars(self):
        return [self.idvar] + self.feature_names


class QCData(BaseData):
    """ data from QC ratings file """

    qcvar = "QC"

    def __init__(self, path_csv: str) -> None:
        super().__init__(path_csv)

    @property
    def required_vars(self):
        return [self.idvar, self.qcvar]


class StudyData(CIVETData, QCData):
    """ data from one CIVET output and one QC ratings file """

    def __init__(self, civet_csv: str, qc_csv: str, cutoff_value: int = 1) -> None:

        self.civet_data = CIVETData(civet_csv)
        self.qc_data = QCData(qc_csv)

        try:
            self.df = pd.merge(self.civet_data.df, self.qc_data.df, on=self.idvar).dropna()
        except Exception as err:
            raise DataFrameMergerError(f"Error merging data from files '{civet_csv}' and '{qc_csv}'") from err

        self.df = self.df[self.required_vars]
        self.df[self.qcvar] = self.df[self.qcvar].apply(pd.to_numeric, errors='coerce')

        if not all(self.df[self.qcvar] >= 0):
            raise NegativeQCRatingError(f"All QC ratings in file {qc_csv} must greater than zero")

        self.cutoff_value = cutoff_value
        if cutoff_value < 1:
            raise ValueError("Cutoff value must be greater than zero")

        self.df[self.qcvar] = np.where(self.df[self.qcvar] < self.cutoff_value, 1, 0)
        assert all([x in range(self.cutoff_value + 1) for x in self.df[self.qcvar]])

    @property
    def required_vars(self):
        return [self.idvar, self.qcvar] + self.feature_names


class MergedData(StudyData):
    """ data from multiple studies merged together """

    def __init__(self, study_paths: list, balanced: bool = False) -> None:

        self.df = None
        for study in study_paths:
            if self.df is None:
                super().__init__(study[0], study[1])
            else:
                study_data = StudyData(study[0], study[1])
                self.df = pd.concat([self.df, study_data.df])

        self.check_required_vars()
        self.check_ids_unique()
        self.df[self.idvar] = list(range(1, len(self.df[self.idvar]) + 1))

        if balanced:
            min_cls = self.df[self.qcvar].value_counts().min()
            self.df = self.df.groupby(self.qcvar).sample(n=min_cls).sort_values(by=self.idvar)

    @property
    def required_vars(self):
        return super().required_vars


class UnprocessedDataset:

    studies_dir = "/Users/joshua/Developer/civetqc/data"

    study_paths = [
        (
            os.path.join(studies_dir, "FEP", "FEP_civet_data.csv"), 
            os.path.join(studies_dir, "FEP", "FEP_QC.csv")
        ),
        (
            os.path.join(studies_dir, "LAM", "LAM_civet_data.csv"), 
            os.path.join(studies_dir, "LAM", "LAM_QC.csv")
        ),
        (
            os.path.join(studies_dir, "INSIGHT", "INSIGHT_civet_data.csv"), 
            os.path.join(studies_dir, "INSIGHT", "INSIGHT_QC.csv")
        ),
        (
            os.path.join(studies_dir, "TOPSY", "TOPSY_civet_data.csv"), 
            os.path.join(studies_dir, "TOPSY", "TOPSY_QC.csv")
        )
    ]

    target_names = ["Acceptable", "Unacceptable"]
        
    def __init__(self) -> None:
        
        data = MergedData(self.study_paths)
        self.feature_names = data.feature_names
        self.features = data.df[data.feature_names].to_numpy()
        self.target = data.df[data.qcvar].to_numpy()
        self.verify_integrity()
        
    def verify_integrity(self):
        assert self.target.ndim == 1 and self.features.ndim == 2
        assert len(self.target) == self.features.shape[0]
        assert len(self.feature_names) == self.features.shape[1]
    
    @property
    def df(self) -> pd.DataFrame:
        df = pd.DataFrame(self.features, columns=self.feature_names)
        df["QC"] = np.where(self.target == 0, self.target_names[0], self.target_names[1])
        return df
