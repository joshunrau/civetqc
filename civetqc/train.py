from __future__ import annotations

import numpy as np

from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE

from scipy.stats.distributions import uniform

from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectPercentile, f_regression
from sklearn.metrics import balanced_accuracy_score, fbeta_score, make_scorer, precision_score, recall_score
from sklearn.model_selection import RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

balanced_accuracy_scorer = make_scorer(balanced_accuracy_score)
f2_scorer = make_scorer(fbeta_score, beta=2)

search = RandomizedSearchCV(
    estimator=Pipeline(
        [
            ("scaler", StandardScaler()),
            ("fs", SelectPercentile(score_func=f_regression)),
            ("sampler", SMOTE()),
            ("clf", None),
        ]
    ),
    param_distributions = [
        {
            "clf": (RandomForestClassifier(),),
            "clf__max_features": uniform(0.01, 0.5),
            "clf__min_samples_split": np.arange(2, 6),
            "clf__min_samples_leaf": np.arange(1, 6),
            "fs__percentile":  [50, 60, 70, 80, 90, 100]
        },
        {
            "clf": (SVC(),),
            "clf__C": uniform(0.1, 10.0),
            "clf__gamma": uniform(0.001, 0.5),
            "clf__kernel": ['linear', 'poly', 'rbf'],
            "fs__percentile": [50, 60, 70, 80, 90, 100]
        },
        
    ],
    cv = 5,
    scoring = {
        "balanced_accuracy": balanced_accuracy_scorer,
        "f2": f2_scorer,
        "precision_acceptable_scans": make_scorer(precision_score, zero_division=1, pos_label=0),
        "recall_acceptable_scans":  make_scorer(recall_score, pos_label=0),
        "precision_unacceptable_scans": make_scorer(precision_score, zero_division=1, pos_label=1),
        "recall_unacceptable_scans":  make_scorer(recall_score, pos_label=1),
    },
    n_iter = 500,
    n_jobs = -1,
    refit = "f2",
)