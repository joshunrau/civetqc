import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from .base import UnprocessedDataset


class DataPartition:
    """ container for test and training sets """
        
    def __init__(self, features: np.ndarray, target: np.ndarray) -> None:
        assert isinstance(features, np.ndarray) and isinstance(target, np.ndarray)
        assert features.ndim == 2 and target.ndim == 1
        self.features, self.target = features, target
    

class Dataset(UnprocessedDataset):
    def __init__(self) -> None:
        super().__init__()
        x_train, x_test, y_train, y_test = train_test_split(
            self.features, self.target, random_state=0)
        self.train = DataPartition(x_train, y_train)
        self.test = DataPartition(x_test, y_test)
        self.verify_integrity()
    
    def __str__(self) -> str:
        
        return f"\n{'-'*79}\n".join([
            self.target_counts,
            self.summary_stats
        ])
    
    @property
    def target_counts(self):
        
        target_counts = {
            "All": [x.tolist() for x in np.unique(self.target, return_counts=True)],
            "Train": [x.tolist() for x in np.unique(self.train.target, return_counts=True)],
            "Test": [x.tolist() for x in np.unique(self.test.target, return_counts=True)],
        }
        
        output_value = ["Target Counts:"]
        for key in target_counts:
            if len(target_counts[key][0]) == len(self.target_names):
                output_value.append(f"\n{key}\n" + "\n".join([f"{name}: {value}" for name, value in zip(self.target_names, target_counts[key][1])]))
            else:
                raise ValueError("Length of target names does not equal number of unique values")
        
        return "\n".join(output_value) + "\n"
    
    @property
    def summary_stats(self):
        mean = self.df.groupby("QC").mean().to_dict()
        std = self.df.groupby("QC").std().to_dict()
        summary_stats = ["Mean and Standard Deviation of Features by QC Rating\n"]
        for feature in self.feature_names:
            feature_stats = [feature]
            for target_name in self.target_names:
                feature_stats.append(f"{target_name}: Mean={mean[feature][target_name]:.2f}, SD={std[feature][target_name]:.2f}")
            summary_stats.append("\n".join(feature_stats) + "\n")
        return "\n".join(summary_stats)