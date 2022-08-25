from __future__ import annotations

from statistics import mean, stdev

import matplotlib.pyplot as plt
import numpy as np

from matplotlib.axes import Axes

from sklearn.inspection import permutation_importance
from sklearn.metrics import fbeta_score, precision_score, recall_score
from sklearn.model_selection import cross_val_predict, StratifiedKFold
from sklearn.pipeline import Pipeline

from .data import CivetData, QCRatingsData
from .model import Model


def plot_discrimination_thresholds(
    pipeline: Pipeline,
    features: np.ndarray,
    target: np.ndarray,
    cv: StratifiedKFold = StratifiedKFold(n_splits=5),
    n_permutations: int = 50,
    ax: Axes | None = None,
) -> Axes:

    discrimination_thresholds = np.arange(0, 1.1, 0.1)

    results: dict[str, list[list[float]]] = {
        "Precision": [[] for x in range(len(discrimination_thresholds))],
        "Recall": [[] for x in range(len(discrimination_thresholds))],
        "F2": [[] for x in range(len(discrimination_thresholds))],
    }

    for _ in range(n_permutations):
        probabilities = cross_val_predict(
            pipeline, features, target, method="predict_proba", cv=cv
        )
        for index, threshold in enumerate(discrimination_thresholds):
            predictions = np.where(probabilities[:, 1] > threshold, 1, 0)
            results["Precision"][index].append(
                precision_score(target, predictions, zero_division=1)
            )
            results["Recall"][index].append(recall_score(target, predictions))
            results["F2"][index].append(fbeta_score(target, predictions, beta=2))

    if ax is None:
        ax = plt.gca()

    for metric, scores in results.items():
        means = [round(mean(score), 2) for score in scores]
        stdevs = [round(stdev(score), 2) for score in scores]
        ax.errorbar(
            discrimination_thresholds, means, label=metric, marker="o", yerr=stdevs
        )

    ax.set_xticks(discrimination_thresholds)
    ax.set_xlabel("Threshold")
    ax.set_ylabel("Scores")
    ax.grid()
    ax.legend()

    return ax


def plot_permutation_importance(
    model: Model,
    civet_data: CivetData,
    qc_data: QCRatingsData,
    ax: Axes,
    n_repeats: int = 25,
    show_title: bool = True,
) -> Axes:
    result = permutation_importance(
        model.clf,
        civet_data.features,
        qc_data.ratings,
        scoring="roc_auc",
        n_repeats=n_repeats,
    )
    sorted_index = result.importances_mean.argsort()
    sorted_names = civet_data.feature_names[sorted_index]
    sorted_importances = result.importances[sorted_index].T
    ax.boxplot(sorted_importances, vert=False, sym="")
    ax.set_xlabel("Decrease in AUC Score")
    ax.set_yticklabels(sorted_names)
    if show_title:
        ax.set_title("Permutation Importance (Testing Data)")
    return ax
