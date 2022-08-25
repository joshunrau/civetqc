from __future__ import annotations

from typing import Any, Callable

from scipy.stats.distributions import uniform

from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectPercentile, f_regression
from sklearn.metrics import (
    balanced_accuracy_score,
    fbeta_score,
    roc_auc_score,
    precision_score,
    recall_score,
    make_scorer,
)
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


class Search(RandomizedSearchCV):
    def __init__(
        self,
        estimator: Pipeline,
        param_distributions: dict[str, list[Any]],
        cv: StratifiedKFold = StratifiedKFold(n_splits=5),
        scoring: dict = {
            "balanced_accuracy": make_scorer(balanced_accuracy_score),
            "f2": make_scorer(fbeta_score, beta=2),
            "precision": make_scorer(precision_score, zero_division=1),
            "recall": make_scorer(recall_score),
            "roc_auc": make_scorer(roc_auc_score),
        },
        n_iter: int = 100,
        n_jobs: int = -1,
        refit: str = "roc_auc",
    ) -> None:
        super().__init__(
            estimator=estimator,
            param_distributions=param_distributions,
            cv=cv,
            scoring=scoring,
            n_iter=n_iter,
            n_jobs=n_jobs,
            refit=refit,
        )

    def get_estimator_name(self) -> str:
        return self.estimator.named_steps.clf.__class__.__name__

    def get_cv_scores(self) -> dict[str, dict[str, float]]:
        metrics = self.__getattribute__("scoring")
        scores: dict[str, dict[str, float]]

        if isinstance(metrics, dict):
            scores = {metric: {} for metric in metrics.keys()}
        else:
            raise TypeError(f"Invalid type: {type(metrics)}")

        for metric in scores.keys():
            i = 0
            while True:
                try:
                    scores[metric][f"Fold {i}"] = round(
                        self.cv_results_[f"split{i}_test_{metric}"][self.best_index_], 3
                    )
                except KeyError:
                    scores[metric]["Mean"] = round(
                        self.cv_results_[f"mean_test_{metric}"][self.best_index_], 3
                    )
                    scores[metric]["SD"] = round(
                        self.cv_results_[f"std_test_{metric}"][self.best_index_], 3
                    )
                    break
                i += 1
        return scores


rfc_search = Search(
    estimator=Pipeline(
        [
            ("scaler", StandardScaler()),
            ("fs", SelectPercentile(score_func=f_regression)),
            ("clf", RandomForestClassifier()),
        ]
    ),
    param_distributions={
        "fs__percentile": [25, 50, 75, 100],
        "clf__n_estimators": [50, 100, 150, 200],
        "clf__max_depth": [5, 10, 20, None],
        "clf__max_features": uniform(0.1, 0.5),
        "clf__min_samples_split": [2, 3, 4],
        "clf__min_samples_leaf": [1, 2, 3],
        # "clf__class_weight": ["balanced"],
    },
)

svc_search = Search(
    estimator=Pipeline(
        [
            ("scaler", StandardScaler()),
            ("fs", SelectPercentile(score_func=f_regression)),
            ("clf", SVC(probability=True)),
        ]
    ),
    param_distributions={
        "fs__percentile": [25, 50, 75, 100],
        "clf__C": uniform(0.1, 10.0),
        "clf__gamma": uniform(0.001, 0.5),
        # "clf__class_weight": ["balanced"],
    },
)
