from sklearn.preprocessing import normalize
import numpy as np

def normalize_columns(X: np.ndarray) -> np.ndarray:
    return normalize(X, axis=0)

def covarianzas_con_precio(X: np.ndarray) -> np.ndarray:
    return np.cov(X)[:, -1][:-1]