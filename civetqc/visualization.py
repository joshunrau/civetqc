from abc import ABC, abstractmethod

import matplotlib.pyplot as plt

from yellowbrick.classifier import ROCAUC
from yellowbrick.model_selection import FeatureImportances

from .models import BaseClassifier


class Figure(ABC):

    @abstractmethod
    def plot(self):
        pass

    def save(self, filepath):
        plt.savefig(filepath, dpi=300, bbox_inches='tight')


class ModelScores(Figure):
    def plot(self, model_names: list, training_scores: list, testing_scores: list, metric_label: str):
        if not all([len(scores) == len(model_names) for scores in [training_scores, testing_scores]]):
            raise ValueError
        fig, axes = plt.subplots(ncols=2, sharey=True, figsize=(10, 4))
        training_plot = axes[0].bar(model_names, training_scores, align='center', alpha=1, color="#408ec6")
        axes[0].bar_label(training_plot, label_type='edge', fmt='%.2f')
        axes[0].set_title("Training")
        axes[0].set_ylabel(metric_label)
        axes[0].set_xticks(list(range(len(model_names))), model_names)
        testing_plot = axes[1].bar(model_names, testing_scores, align='center', alpha=1, color="#1e2761")
        axes[1].bar_label(testing_plot, label_type='edge', fmt='%.2f')
        axes[1].set_title("Testing")
        axes[1].set_xticks(list(range(len(model_names))), model_names)
        fig.subplots_adjust(top=1, wspace=.05)

class ROCCurve:
    def plot(self, clf: BaseClassifier, X_train, X_test, y_train, y_test):
        if not isinstance(clf, BaseClassifier):
            raise TypeError(f"Classifier must inherit from {BaseClassifier}")
        viz = ROCAUC(clf.best_estimator_, binary=True, title=" ")
        viz.fit(X_train, y_train)
        viz.score(X_test, y_test)
        viz.finalize()