import numpy as np

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