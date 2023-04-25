from typing import List
import numpy as np

def discounted_cumulative_gain(relevance: List[float], k: int, method: str = "standard") -> float:
    dcg = 0
    relevance_considered = relevance[:k]
    for order in range(1, len(relevance_considered)+1):
        i = order - 1
        if method == "standard":
            dcg_i = relevance_considered[i] / np.log2(order+1)
        else:
            dcg_i = (2**relevance_considered[i] - 1) / np.log2(order+1)
        dcg = dcg + dcg_i
    return dcg

def normalized_dcg(relevance: List[float], k: int, method: str = "standard") -> float:
    """Normalized Discounted Cumulative Gain.

    Parameters
    ----------
    relevance : `List[float]`
        Video relevance list
    k : `int`
        Count relevance to compute
    method : `str`, optional
        Metric implementation method, takes the values
        `standard` - adds weight to the denominator
        `industry` - adds weights to the numerator and denominator
        `raise ValueError` - for any value

    Returns
    -------
    score : `float`
        Metric score
    """
    relevances_sorted = list(np.sort(relevance)[::-1])
    # dcg:
    dcg = discounted_cumulative_gain(relevance, k, method = method)
    # dcg_ideal:
    dcg_ideal = discounted_cumulative_gain(relevances_sorted, k, method = method)
    score = dcg / dcg_ideal
    return score