import numpy as np

def non_zero_columns(coeffs: np.ndarray, tol=1e-12):
    matrix = np.asarray(coeffs)
    return list(np.where(np.any(np.abs(matrix) > tol, axis=0))[0])