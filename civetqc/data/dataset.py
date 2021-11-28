import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from imblearn.over_sampling import SMOTE
from sklearn.decomposition import PCA
from sklearn.manifold import Isomap
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from .base import BaseData, StudyData, MergedData
from .studies import Studies

class Dataset:

    def __init__(self, data: BaseData = MergedData()):
        self.features = data.df[data.feature_names].to_numpy()
        self.feature_names = data.feature_names
        self.target = data.df[data.qcvar].to_numpy()
        self.target_names = data.target_names

    @property
    def n_features(self):
        return self.features.shape[1]

    @property
    def n_samples(self):
        return self.target.shape[0]

    @property
    def means(self):
        return self.get_statistic_by_target(np.mean)

    @property
    def stds(self):
        return self.get_statistic_by_target(np.std)

    @property
    def train(self):
        x_train, _, y_train, _ = train_test_split(self.features, self.target, random_state=0)
        return {
            "features": x_train,
            "target": y_train
        }

    @property
    def test(self):
        _, x_test, _, y_test = train_test_split(self.features, self.target, random_state=0)
        return {
            "features": x_test,
            "target": y_test
        }

    # Summary statistics

    def get_target_counts(self):

        target_counts = [x.tolist() for x in np.unique(self.target, return_counts=True)]

        output_value = [f"Target Counts"] + [f"{name}: {value}" for name, value in zip(self.target_names, target_counts[1])]

        if len(target_counts[0]) != len(self.target_names):
            raise ValueError(f"Length of target names does not equal number of unique values")

        return "\n".join(output_value) + "\n"

    def get_statistic_by_target(self, f):

        features_by_target = {
            self.target_names[0]: self.features[self.target == 0],
            self.target_names[1]: self.features[self.target == 1]
        }

        raw = {name: f(features_by_target[name], axis=0).round(3) for name in self.target_names}
        formatted = {feature: {} for feature in self.feature_names}
        for i in range(len(self.feature_names)):
            for target in self.target_names:
                formatted[self.feature_names[i]][target] = raw[target][i]
        return formatted

    # Balancing
    def apply_smote(self):
        model = SMOTE(random_state=0)
        self.features, self.target = model.fit_resample(self.features, self.target)

    # Preprocessing
    def apply_std_scaler(self, **kwargs):
        model = StandardScaler(**kwargs)
        model.fit(self.features)
        self.features = model.transform(self.features)

    # Dimensionality reduction

    def apply_pca(self, **kwargs):
        if kwargs == {}:
            kwargs = {'n_components': 2, 'random_state': 0}
        model = PCA(**kwargs)
        model.fit(self.features)
        self.features = model.transform(self.features)
        self.feature_names = np.array([f"PCA{x}" for x in range(1, model.n_components_ + 1)])

    def apply_isomap(self, **kwargs):
        if kwargs == {}:
            kwargs = {'n_components': 2}
        model = Isomap(**kwargs)
        model.fit(self.features)
        self.features = model.transform(self.features)
        self.feature_names = np.array([f"ISO{x}" for x in range(1, model.n_components + 1)])

    # Plots

    def scatterplot(self):
        if self.n_features != 2:
            raise ValueError(f"Number of features must be 2, not {self.n_features}")
        return sns.scatterplot(data=self.features)

    def plot_distribution(self) -> None:

        plot_data = pd.DataFrame(np.hstack([self.features, self.target[:, np.newaxis]]),
                                 columns=self.feature_names.tolist() + ["QC"])

        rows_needed = len(self.feature_names) // 2 + len(self.feature_names) % 2

        fig, axes = plt.subplots(rows_needed, 2, figsize=(8, rows_needed * 2))
        ax = axes.ravel()

        for i in range(len(self.feature_names)):
            sns.kdeplot(data=plot_data, x=self.feature_names[i], hue="QC", fill=True, common_norm=False, bw_adjust=1,
                        alpha=.5, ax=ax[i])

            x_label = "\n".join([
                self.feature_names[i],
                "Mean: " + ", ".join(
                    [f"{x}: {self.means[self.feature_names[i]][x]}" for x in self.means[self.feature_names[i]]]),
                "SD: " + ", ".join(
                    [f"{x}: {self.stds[self.feature_names[i]][x]}" for x in self.stds[self.feature_names[i]]])
            ])

            ax[i].set_xlabel(x_label)

        fig.tight_layout(h_pad=2)
        fig.set_dpi(300)

    @classmethod
    def datasets_by_study(cls):
        studies = {}
        for name, filepaths in Studies.filepaths.items():
            studies[name] = cls(data = StudyData(filepaths[0], filepaths[1]))
        return studies

