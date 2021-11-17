from abc import ABC, abstractmethod

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from ..exceptions import VariableNotFoundError, DuplicateIdentifierError, DataFrameMergerError, NegativeQCRatingError


class BaseData(ABC):
    """ inherited by all data classes """

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
    """ encapsulates data from one CIVET output and one QC ratings file """

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

        self.df[self.qcvar] = np.where(self.df[self.qcvar] < self.cutoff_value, 0, 1)
        assert all([x in range(self.cutoff_value + 1) for x in self.df[self.qcvar]])

    @property
    def required_vars(self):
        return [self.idvar, self.qcvar] + self.feature_names


class MergedDataset(StudyData):

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


class DataPartition:
    """ container for test and training sets """

    def __init__(self, features: np.ndarray, target: np.ndarray) -> None:
        assert isinstance(features, np.ndarray) and isinstance(target, np.ndarray)
        assert features.ndim == 2 and target.ndim == 1
        self.features, self.target = features, target

    def __str__(self) -> str:
        return "Target Class Counts\n" + '\n'.join(
            f"{': '.join([str(y) for y in list(x)])} ({round(x[-1] / len(self.target) * 100, 2)}%)" for x in
            np.array(np.unique(self.target, return_counts=True)).T)


class Dataset:
    """ final format of dataset to build model """

    target_names = ["Acceptable", "Unacceptable"]

    def __init__(self, data: MergedDataset) -> None:
        self.feature_names = data.feature_names
        self.features = data.df[data.feature_names].to_numpy()
        self.target = data.df[data.qcvar].to_numpy()
        x_train, x_test, y_train, y_test = train_test_split(self.features, self.target, random_state=0)
        self.train = DataPartition(x_train, y_train)
        self.test = DataPartition(x_test, y_test)
