from typing import Tuple
from concurrent.futures import ProcessPoolExecutor
import numpy as np
from sklearn.base import ClassifierMixin
from sklearn.metrics import roc_auc_score

def get_percentile_ci(bootstrap_stats, alpha):
    """build percentile confidence interval"""
    left = np.quantile(bootstrap_stats, alpha/2)
    right = np.quantile(bootstrap_stats, 1 - alpha/2)
    return left, right

def compute_score(random_indices, X, y, classifier):
    X_subset = X[random_indices, :]
    y_subset = y[random_indices]
    if len(set(y_subset)) < 2: # otherwise roc_auc is not determined
        return None
    else:
        return roc_auc_score(y_subset, classifier.predict_proba(X_subset)[:, 1])

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
    random_indices_list = list(np.random.choice(np.arange(0, n), (n_bootstraps, n), True))
    
    bootstrap_scores = []
    with ProcessPoolExecutor() as executor:
        futures = [executor.submit(compute_score, random_indices, X, y, classifier) 
                   for random_indices in random_indices_list]
        
        for future in futures:
            score = future.result()
            if score is not None:
                bootstrap_scores.append(score)

    return get_percentile_ci(bootstrap_scores, alpha)
