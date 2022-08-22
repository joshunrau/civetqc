from __future__ import annotations

import numpy as np

from scipy.stats.distributions import loguniform

from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

cv = StratifiedKFold(n_splits=5)

rfc_search = RandomizedSearchCV(
  estimator = Pipeline([
    ('scaler', StandardScaler()),
    ('clf', RandomForestClassifier())
]),
  param_distributions = {
    'clf__n_estimators': np.arange(50, 501),
    'clf__max_depth': np.arange(5, 51),
    'clf__max_features': loguniform(0.01, 1.0),
    'clf__min_samples_split': np.arange(2, 6),
    'clf__min_samples_leaf': np.arange(1, 6),
  },
  cv=cv,
  scoring='roc_auc',
  n_iter=50
)

gbc_search = RandomizedSearchCV(
  estimator=Pipeline([
    ('scaler', StandardScaler()),
    ('clf', GradientBoostingClassifier()),
  ]),
  param_distributions={
    "clf__learning_rate": loguniform(0.001, 0.5),
    "clf__min_samples_split": loguniform(0.1, 0.9),
    "clf__min_samples_leaf": loguniform(0.1, 0.5),
    "clf__max_depth": np.arange(2, 9),
    "clf__max_features": ["log2","sqrt"],
    "clf__criterion": ["friedman_mse",  "squared_error"],
    "clf__subsample": loguniform(0.5, 1)
    },
  cv=cv,
  scoring='roc_auc',
  n_iter=50
)

svc_search = RandomizedSearchCV(
  estimator=Pipeline([
    ('scaler', StandardScaler()),
    ('clf', SVC()),
  ]),
  param_distributions={
    'clf__C': loguniform(0.01, 1000.0),
    'clf__gamma': loguniform(0.0001, 10.0),
    'clf__degree': np.arange(1, 4),
    'clf__kernel': ['linear', 'poly', 'rbf'],
    'clf__probability': [True],
    'clf__class_weight': [None, 'balanced']
  },
  cv=cv,
  scoring='roc_auc',
  n_iter=50
)

def train(X: np.ndarray, y: np.ndarray) -> list[RandomizedSearchCV]:
  searches = [rfc_search, gbc_search, svc_search]
  for search in searches:
    search.fit(X, y)
  return searches

def get_estimator_name(search: RandomizedSearchCV) -> str:
    return search.estimator.named_steps.clf.__class__.__name__

def get_scores(searches: list[RandomizedSearchCV]) -> dict:
    scores = {}
    for search in searches:
        scores[get_estimator_name(search)] = round(search.best_score_, 2)
    return scores
  
def get_cv_scores(searches: list[RandomizedSearchCV]) -> dict:
    cv_scores = {}
    for search in searches:
        key = get_estimator_name(search)
        cv_scores[key] = []
        best_index = search.cv_results_['rank_test_score'].tolist().index(1)
        i = 0
        while True: ## So this will work with any number of splits
            try:
                score = search.cv_results_[f'split{i}_test_score'][best_index]
            except KeyError:
                break
            cv_scores[key].append(round(score, 2))
            i += 1
    return cv_scores
