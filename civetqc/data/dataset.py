import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

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
        self.verify_integrity()
    
    @property
    def features_by_target(self):
        return {
            self.target_names[0] : self.features[self.target == 0],
            self.target_names[1] : self.features[self.target == 1]
        }

    @property
    def train(self):
        x_train, _, y_train, _ = train_test_split(self.features, self.target, random_state=0)
        return {"features": x_train, "target": y_train}

    @property
    def test(self):
        _, x_test, _, y_test = train_test_split(self.features, self.target, random_state=0)
        return {"features": x_test, "target": y_test}

    @property
    def means(self):
        return self.get_statistic_by_target(np.mean)
    
    @property
    def stds(self):
        return self.get_statistic_by_target(np.std)
    
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
        self.features = np.vstack([self.train["features"], self.test["features"] ])
    
    def apply_standard_scaler(self):
        scaler = StandardScaler()
        scaler.fit(self.features)
        self.features = scaler.transform(self.features)
    
    def plot_distribution(self) -> None:

        def get_x_label(self, i: int) -> str:
            return "\n".join([
                self.feature_names[i],
                "Mean: " + ", ".join([f"{x}: {self.means[self.feature_names[i]][x]}" for x in self.means[self.feature_names[i]]]),
                "SD: " + ", ".join([f"{x}: {self.stds[self.feature_names[i]][x]}" for x in self.stds[self.feature_names[i]]])
            ])
        
        plot_data = pd.DataFrame(np.hstack([self.features, self.target[:, np.newaxis]]), columns = self.feature_names + ["QC"])

        fig, axes = plt.subplots(15, 2, figsize=(20, 70))
        ax = axes.ravel()

        for i in range(29):
            sns.kdeplot(data=plot_data, x=self.feature_names[i], hue="QC", fill=True, common_norm=False, bw_adjust=1, alpha=.5, ax=ax[i])
            ax[i].set_xlabel(get_x_label(self, i))
        
        fig.tight_layout(h_pad=2)
        fig.set_dpi(300)
