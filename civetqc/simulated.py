from abc import ABC, abstractmethod
from .dataset import Dataset
import os

import numpy as np
import pandas as pd


class SimulatedData(ABC):

    simulated_dir = "/Users/joshua/Developer/civetqc/data/simulated"

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


class SimulatedDataset(SimulatedData, Dataset):

    def __init__(self) -> None:
        super().__init__(cutoff_value=1, balanced=True, list_features=None)
    
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
    def simulated_data(self):
        data = {v: [] for v in self.required_all_vars}
        data[self.idvar] += list(range(100))
        data[self.qcvar] += [np.random.randint(0, 2) for x in range(100)]
        for feature in self.feature_names:
            for qc in data[self.qcvar]:
                m, sd = self.means[feature][qc], self.stds[feature][qc]
                data[feature].append(np.random.normal(m, sd))
        return pd.DataFrame(data)
    
    def duplicate_id(self):
        self.simulated_data.loc[0, self.idvar] = self.simulated_data.loc[1, self.idvar]


class SimulatedCIVETData(SimulatedDataset):

    def __init__(self) -> None:
        super().__init__()

    @property
    def output_path(self):
        return os.path.join(self.simulated_dir, "simulated_civet.csv")

    @property
    def simulated_data(self):
        return super().simulated_data[self.required_civet_vars]
    

class SimulatedQCData(SimulatedDataset):
    
    def __init__(self) -> None:
        super().__init__()
    
    @property
    def output_path(self):
        return os.path.join(self.simulated_dir, "simulated_qc.csv")
    
    @property
    def simulated_data(self):
        return super().simulated_data[self.required_user_vars]
    
    def remove_qc_var(self):
        self.simulated_data = self.simulated_data.drop([self.qcvar], axis=1)

    def negative_qc_rating(self):
        self.simulated_data.loc[0, self.qcvar] = -1