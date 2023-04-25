# Average nDCG function:
from typing import List

import numpy as np

def discounted_cumulative_gain(relevance: List[float], k: int, method: str = "standard") -> float:
    """calculates DCG for list"""
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
    """calculated nDCG for list"""
    relevances_sorted = list(np.sort(relevance)[::-1])
    # dcg:
    dcg = discounted_cumulative_gain(relevance, k, method = method)
    # dcg_ideal:
    dcg_ideal = discounted_cumulative_gain(relevances_sorted, k, method = method)
    score = dcg / dcg_ideal
    return score

def avg_ndcg(list_relevances: List[List[float]], k: int, method: str = 'standard') -> float:
    """Average nDCG

    Parameters
    ----------
    list_relevances : `List[List[float]]`
        Video relevance matrix for various queries
    k : `int`
        Count relevance to compute
    method : `str`, optional
        Metric implementation method, takes the values ​​\
        `standard` - adds weight to the denominator\
        `industry` - adds weights to the numerator and denominator\
        `raise ValueError` - for any value

    Returns
    -------
    score : `float`
        Metric score
    """
    queries_number = len(list_relevances)
    score = 0
    for query in list_relevances:
        nDCG_iter = normalized_dcg(query, k, method)
        score = score + nDCG_iter
    score = score / queries_number

    return score
