from typing import Tuple

import numpy as np
from sklearn.base import ClassifierMixin
from sklearn.metrics import roc_auc_score
from sklearn.utils import resample

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
    alpha = 1 - conf
    bootstrap_scores = []
    for _ in range(n_bootstraps):
        X_sample, y_sample = resample(X, y, stratify=y)
        score = roc_auc_score(y_sample, classifier.predict_proba(X_sample)[:, 1])
        bootstrap_scores.append(score)

    return get_percentile_ci(bootstrap_scores, alpha)
