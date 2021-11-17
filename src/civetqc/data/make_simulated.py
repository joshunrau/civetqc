import os
from abc import ABC, abstractmethod

import numpy as np
import pandas as pd

from .create_dataset import MergedDataset


class SimulatedData(ABC):

    @property
    @abstractmethod
    def simulated_data(self):
        pass

    @property
    @abstractmethod
    def output_path(self):
        pass

    def to_csv(self):
        self.simulated_data.to_csv(self.output_path, index=False)


class SimulatedDataset(SimulatedData, MergedDataset):

    def __init__(self, study_paths: list, balanced: bool, simulated_dir: str) -> None:
        super().__init__(study_paths, balanced)
        self.simulated_dir = simulated_dir

    @property
    def output_path(self):
        return os.path.join(self.simulated_dir, "simulated_dataset.csv")

    @property
    def means(self):
        return self.df.groupby(self.qcvar).mean().to_dict()

    @property
    def stds(self):
        return self.df.groupby(self.qcvar).std().to_dict()

    @property
    def required_vars(self):
        return super().required_vars

    @property
    def simulated_data(self):
        data = {v: [] for v in self.required_vars}
        data[self.idvar] += list(range(100))
        data[self.qcvar] += np.random.randint(0, 2, size=100).tolist()
        for feature in self.feature_names:
            for qc in data[self.qcvar]:
                m, sd = self.means[feature][qc], self.stds[feature][qc]
                data[feature].append(np.random.normal(m, sd))
        return pd.DataFrame(data)

    def duplicate_id(self):
        self.simulated_data.loc[0, self.idvar] = self.simulated_data.loc[1, self.idvar]


class SimulatedCIVETData(SimulatedDataset):

    def __init__(self, study_paths: list, balanced: bool, simulated_dir: str) -> None:
        super().__init__(study_paths, balanced, simulated_dir)

    @property
    def output_path(self):
        return os.path.join(self.simulated_dir, "simulated_civet.csv")

    @property
    def simulated_data(self):
        return super().simulated_data[[self.idvar] + self.feature_names]


class SimulatedQCData(SimulatedDataset):

    def __init__(self, study_paths: list, balanced: bool, simulated_dir: str) -> None:
        super().__init__(study_paths, balanced, simulated_dir)

    @property
    def output_path(self):
        return os.path.join(self.simulated_dir, "simulated_qc.csv")

    @property
    def simulated_data(self):
        return super().simulated_data[[self.idvar, self.qcvar]]

    def remove_qc_var(self):
        self.simulated_data = self.simulated_data.drop([self.qcvar], axis=1)

    def negative_qc_rating(self):
        self.simulated_data.loc[0, self.qcvar] = -1
