import numpy as np

def turnover_error(y_true: np.array, y_pred: np.array) -> float:
    val1 = 0.75 * np.abs(y_true - y_pred) # 0.25 - gamma
    val2 = (1-0.75) * np.abs(y_true - y_pred) # 0.25 - gamma
    q_loss = np.where(y_true >= y_pred, val1, val2)
    error = np.mean(q_loss)
    return error