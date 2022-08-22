from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from matplotlib.axes import Axes
from sklearn.metrics import fbeta_score, precision_score, recall_score

from sklearn.base import BaseEstimator

def plot_discrimination_thresholds(estimator: BaseEstimator, features: np.ndarray, target: np.ndarray, ax: Axes | None = None) -> Axes:
    discrimination_thresholds = np.arange(0, 1.1, .1)
    scores = {
        'Precision': [],
        'Recall': [],
        'F2': []
    }
    for threshold in discrimination_thresholds:
        probabilities = estimator.predict_proba(features)
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
    ax.set_title("Discrimination Thresholds")
    ax.legend()
    
    return ax