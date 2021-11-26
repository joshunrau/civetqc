import os

import numpy as np
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split

from .base import MergedData


class Dataset:
    
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
        self.df = data.df
        self.feature_names = data.feature_names
        self.features = data.df[data.feature_names].to_numpy()
        self.target = data.df[data.qcvar].to_numpy()
        
        self.features_by_target = {
            self.target_names[0] : self.features[self.target == 0],
            self.target_names[1] : self.features[self.target == 1]
        }
        
        x_train, x_test, y_train, y_test = train_test_split(self.features, self.target, random_state=0)

        self.train = {
            "features": x_train,
            "target": y_train
        }

        self.test = {
            "features": x_test,
            "target": y_test
        }

        self.means = self.get_statistic_by_target(np.mean)
        self.stds = self.get_statistic_by_target(np.std)

        self.verify_integrity()
        
    def __str__(self) -> str:

        return f"\n{'-' * 79}\n".join([
            self.get_target_counts(),
            self.get_summary_stats()
        ])
    
    def verify_integrity(self):
        assert self.target.ndim == 1 and self.features.ndim == 2
        assert len(self.target) == self.features.shape[0]
        assert len(self.feature_names) == self.features.shape[1]

    def get_target_counts(self):

        target_counts = {
            "All": [x.tolist() for x in np.unique(self.target, return_counts=True)],
            "Train": [x.tolist() for x in np.unique(self.train['target'], return_counts=True)],
            "Test": [x.tolist() for x in np.unique(self.test['target'], return_counts=True)],
        }

        output_value = ["Target Counts:"]
        for key in target_counts:
            if len(target_counts[key][0]) == len(self.target_names):
                output_value.append(f"\n{key}\n" + "\n".join(
                    [f"{name}: {value}" for name, value in zip(self.target_names, target_counts[key][1])]))
            else:
                raise ValueError("Length of target names does not equal number of unique values")

        return "\n".join(output_value) + "\n"
    
    def get_statistic_by_target(self, f):
        raw = {name: f(self.features_by_target[name], axis=0).round(3) for name in self.target_names}
        formatted = {feature: {} for feature in self.feature_names}
        for i in range(len(self.feature_names)):
            for target in self.target_names:
                formatted[self.feature_names[i]][target] = raw[target][i]
        return formatted
    
    def get_summary_stats(self):
        summary_stats = ["Mean and Standard Deviation of Features by QC Rating\n"]
        for feature in self.feature_names:
            feature_stats = [feature]
            for name in self.target_names:
                feature_stats.append(f"{name}: Mean={self.means[feature][name]:.2f}, SD={self.stds[feature][name]:.2f}")
            summary_stats.append("\n".join(feature_stats) + "\n")
        return "\n".join(summary_stats)

    def over_sample(self):
        est = SMOTE(random_state=0)
        self.train["features"], self.train["target"] = est.fit_resample(self.train["features"], self.train["target"])