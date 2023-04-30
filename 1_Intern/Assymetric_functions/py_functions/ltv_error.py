import numpy as np

def ltv_error(y_true: np.array, y_pred: np.array) -> float:
    """Quantile loss function for LTV prediction model"""
    val1 = 0.1 * np.abs(y_true - y_pred) # 0.1 - gamma
    val2 = (1-0.1) * np.abs(y_true - y_pred) # 0.1 - gamma
    q_loss = np.where(y_true >= y_pred, val1, val2)
    error = np.mean(q_loss)
    return error
