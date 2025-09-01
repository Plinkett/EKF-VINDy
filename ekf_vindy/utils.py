import numpy as np

def find_non_zero(coeffs: np.ndarray, tol=1e-12):
    matrix = np.asarray(coeffs)
    return [list(np.where(np.abs(row) > tol)[0]) for row in matrix]