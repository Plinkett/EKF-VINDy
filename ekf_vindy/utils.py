import numpy as np
from typing import Callable

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
