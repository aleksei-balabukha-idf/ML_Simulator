from typing import Tuple

import numpy as np
from sklearn.base import ClassifierMixin
from sklearn.metrics import roc_auc_score

def get_percentile_ci(bootstrap_stats, alpha):
    """build percentile confidence interval"""
    quantiles = np.quantile(bootstrap_stats, [alpha/2, 1 - alpha/2])
    return quantiles[0], quantiles[1]

def roc_auc_ci(
    classifier: ClassifierMixin,
    X: np.ndarray,
    y: np.ndarray,
    conf: float = 0.95,
    n_bootstraps: int = 10_000,
) -> Tuple[float, float]:
    """Returns confidence bounds of the ROC-AUC"""
    n = len(X)
    alpha = 1 - conf
    random_indices = np.random.choice(np.arange(0, n), (n_bootstraps, n), replace=True)
    bootstrap_scores = np.zeros(n_bootstraps)
    for i, indices in enumerate(random_indices):
        X_subset = X[indices, :]
        y_subset = y[indices]
        if len(np.unique(y_subset)) > 1:
            score_iter = roc_auc_score(y_subset, classifier.predict_proba(X_subset)[:, 1])
            bootstrap_scores[i] = score_iter
    return get_percentile_ci(bootstrap_scores, alpha)
