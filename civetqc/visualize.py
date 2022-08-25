from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from matplotlib.axes import Axes

from sklearn.inspection import permutation_importance
from sklearn.metrics import fbeta_score, precision_score, recall_score
from sklearn.model_selection import GridSearchCV

from .data import CivetData, QCRatingsData
from .model import Model
from .train import get_estimator_name

def plot_discrimination_thresholds(search: GridSearchCV, features: np.ndarray, target: np.ndarray, ax: Axes | None = None, show_title: bool = True) -> Axes:
    discrimination_thresholds = np.arange(0, 1.1, .1)
    scores = {
        'Precision': [],
        'Recall': [],
        'F2': []
    }
    for threshold in discrimination_thresholds:
        probabilities = search.predict_proba(features)
        if probabilities.shape[1] != 2: # In case later someone adds borderline fails
          raise ValueError('Not a binary classification task!')
        predictions = np.where(probabilities[:, 1] > threshold, 1, 0)
        scores['Precision'].append(precision_score(target, predictions, zero_division=1))
        scores['Recall'].append(recall_score(target, predictions))
        scores['F2'].append(fbeta_score(target, predictions, beta=2))
    
    if ax is None:
      ax = plt.gca()
    
    for key, value in scores.items():
        ax.plot(discrimination_thresholds, value, label=key, marker='o')
    ax.set_xticks(discrimination_thresholds)
    ax.set_xlabel("Threshold")
    ax.set_ylabel("Scores")
    ax.legend()
    if show_title:
        ax.set_title(f"Discrimination Thresholds for {get_estimator_name(search)}")
    return ax

def plot_permutation_importance(model: Model, civet_data: CivetData, qc_data: QCRatingsData, ax: Axes, n_repeats: int = 25, show_title: bool = True) -> Axes:
    result = permutation_importance(
        model.clf, 
        civet_data.features, 
        qc_data.ratings, 
        scoring='roc_auc', 
        n_repeats=n_repeats
    )
    sorted_index = result.importances_mean.argsort()
    sorted_names = civet_data.feature_names[sorted_index]
    sorted_importances = result.importances[sorted_index].T
    ax.boxplot(sorted_importances, vert=False, sym='')
    ax.set_xlabel('Decrease in AUC Score')
    ax.set_yticklabels(sorted_names)
    if show_title:
        ax.set_title('Permutation Importance (Testing Data)')
    return ax