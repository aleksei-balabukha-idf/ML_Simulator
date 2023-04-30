from typing import Tuple

import numpy as np
from sklearn.base import ClassifierMixin
from sklearn.metrics import roc_auc_score

def get_percentile_ci(bootstrap_stats, alpha):
    """build percentile confidence interval"""
    left = np.quantile(bootstrap_stats, alpha/2)
    right = np.quantile(bootstrap_stats, 1 - alpha/2)
    return left, right



def roc_auc_ci(
    classifier: ClassifierMixin,
    X: np.ndarray,
    y: np.ndarray,
    conf: float = 0.95,
    n_bootstraps: int = 10_000,
) -> Tuple[float, float]:
    """Returns confidence bounds of the ROC-AUC"""
    # number of records:
    n = len(X)
    # confidence interval size:
    alpha = 1 - conf
    # generate random subset from initial data n_bootstraps times:
    bootstrap_scores = []
    random_indices_list = list(np.random.choice(np.arange(0, n), (n_bootstraps, n), True))
    for random_indices in random_indices_list:
        X_subset = X[random_indices, :]
        y_subset = y[random_indices]
        # calculate roc_auc_score n_bootstraps times
        if len(set(y_subset)) < 2: # otherwise roc_auc is not determined
            pass
        else:
            score_iter = roc_auc_score(y_subset, classifier.predict_proba(X_subset)[:, 1])
            bootstrap_scores.append(score_iter)
    # calculate percentile confidence interval:
    percentile_ci = get_percentile_ci(bootstrap_scores, alpha)
    lcb = percentile_ci[0]
    ucb = percentile_ci[1]

    return (lcb, ucb)
