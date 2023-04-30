# DCG python function:
from typing import List

import numpy as np


def discounted_cumulative_gain(relevance: List[float], k: int, method: str = "standard") -> float:
    """Discounted Cumulative Gain

    Parameters
    ----------
    relevance : `List[float]`
        Video relevance list
    k : `int`
        Count relevance to compute
    method : `str`, optional
        Metric implementation method, takes the values​​
        `standard` - adds weight to the denominator
        `industry` - adds weights to the numerator and denominator
        `raise ValueError` - for any value

    Returns
    -------
    score : `float`
        Metric score
    """
    score = 0
    relevance_considered = relevance[:k]
    for order in range(1, len(relevance_considered)+1):
        i = order - 1
        if method == "standard":
            dcg_i = relevance_considered[i] / np.log2(order+1)
        else:
            dcg_i = (2**relevance_considered[i] - 1) / np.log2(order+1)
        score = score + dcg_i

    return score
