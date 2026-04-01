import numpy as np
from typing import Callable, List

def non_zero_columns(coeffs: np.ndarray, tol=1e-12):
    """
    Select columns i.e., terms of interest (from the library) that have at least one non-zero element
    """
    matrix = np.asarray(coeffs)
    return list(np.where(np.any(np.abs(matrix) > tol, axis=0))[0])

def add_noise_with_snr(signal: np.ndarray, snr: float):
    """
    The signal-to-noise ratio is here defined as the ratio between the mean square of the signal, and the mean square of the noise component
    The snr argument is NOT expressed in decibels. The p's in the code refer to "power".
    """
    
    p_signal = np.mean(np.square(signal))
    p_signal_dB = 10 * np.log10(p_signal)
    p_noise_dB  = p_signal_dB - snr
    signal_dev  = 10 ** (p_noise_dB / 10)

    noise = np.random.normal(0, np.sqrt(signal_dev), signal.shape)
    noisy_signal = signal + noise

    return noisy_signal

def integration_step(y: np.ndarray, f: Callable, dt: float, method='Euler'):
    """
    A simple integration step (Euler or RK4). We assume an autonomous ODE i.e., no explicit time-dependence.
    This is for either evolving the state and/or solving the Lyapunov equation to obtain the predicted covariance.
    It is worth noting that we are assuming our linearization (through the Jacobian) holds for the intermediate steps of the RK4. 
    Ideally, you would compute the Jacobian at the intermediate states as well.
    """
    if method in ('Euler', 'FE', 'EF'):
        y_new = y + dt * f(y)
    elif method == 'RK4':
        y_k1 = f(y)
        y_k2 = f(y + dt/2 * y_k1)
        y_k3 = f(y + dt/2 * y_k2)
        y_k4 = f(y + dt * y_k3)
        y_new = y + dt/6 * (y_k1 + 2 * y_k2 + 2 * y_k3 + y_k4)
    else:
        raise ValueError("Integration method not supported: " + str(method))
    return y_new

def extract_tracked_entries(matrix: np.ndarray, tracked_terms: List[List[int]]) -> np.ndarray:
    """
    Receives a matrix of Gaussian variances (transformed from the Laplace posterior scales) and extracts the variances according to tracked_terms,
    the mechanism is the same as the one we use to track the coefficients in the EKF class. 
    
    Returns a single vector with all the variances, which will then be part of the diagonal prior covariance matrix.
    """
    values = []
    for i, row in enumerate(tracked_terms):
        for col in row:
            values.append(matrix[i, col])
    return np.asarray(values)

def scale_to_var_optimal(scales: np.ndarray):
    """
    We take a matrix of scales from the Laplace posteriors, and we compute the variance of the
    Gaussians that minimize the KL divergence w.r.t. such Laplace (jointly with the mean). 
    
    The optimal variance (in the KL sense) is 2 * loc^2 / pi.
    """
    variances = 2 * scales ** 2 / np.pi 
    return variances.T