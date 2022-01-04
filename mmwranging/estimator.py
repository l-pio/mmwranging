import numpy as np
from math import floor, ceil
from scipy.signal import windows
from scipy.optimize import curve_fit


def basic_estimator(magn_response, phase_response=None, roi=None):
    """Estimate pulse position and pulse phase in radar echo without interpolation."""
    # Peak detection
    x = np.arange(magn_response.shape[-1])
    masc = (x >= roi[0]) * (x <= roi[1]) if roi is not None else x <= magn_response.shape[-1] // 2
    n = np.argmax(magn_response * masc, axis=-1)

    # Phase
    if phase_response is not None:
        phi = {
            1: lambda: phase_response[n],
            2: lambda: np.array([phase_response[idx, _n] for idx, _n in enumerate(n)])
            }[np.ndim(phase_response)]()
        return n, phi,
    else:
        return n


def qi_estimator(magn_response, phase_response=None, roi=None):
    """Estimate pulse position and pulse phase in radar echo using quadratic interpolation (QI)."""
    # Peak detection
    x = np.arange(magn_response.shape[-1])
    masc = (x >= roi[0]) * (x <= roi[1]) if roi is not None else x <= magn_response.shape[-1] // 2
    n = np.argmax(magn_response * masc, axis=-1)

    # Interpolation
    n_i = qi(magn_response, n)

    return (n_i, linear_phase_interpolation(phase_response, n_i),) if phase_response is not None else n_i


def qips_estimator(window, magn_response, phase_response=None, roi=None):
    """Estimate pulse position and pulse phase in power-scaled radar echo using quadratic interpolation (QIPS)."""
    # Peak detection
    x = np.arange(magn_response.shape[-1])
    masc = (x >= roi[0]) * (x <= roi[1]) if roi is not None else x <= magn_response.shape[-1] // 2
    n = np.argmax(magn_response * masc, axis=-1)

    # Interpolation
    p = {
        'triang': 0.23,
        'parzen': 0.10,
        'bohman': 0.14,
        'blackman': 0.13,
        'nuttall': 0.08,
        'blackmanharris': 0.09,
        'flattop': 1.00,
        'bartlett': 0.23,
        'hanning': 0.23,
        'barthann': 0.22,
        'hamming': 0.19,
        'cosine': 0.37,
        'hann': 0.23,
        'exponential': 0.50,
        'tukey': 0.50,
        'taylor': 0.31,
    }[window]
    n_i = qips(magn_response, n, p)

    return (n_i, linear_phase_interpolation(phase_response, n_i),) if phase_response is not None else n_i


def qips(magn_response, n, p):
    """Quadratic interpolation in power-scaled radar echo (QIPS)."""
    if np.ndim(magn_response) == 1:
        # Single-dimensional array
        n_i = (n - (magn_response[n]**p - magn_response[n + 1]**p) /
               (2 * magn_response[n]**p - magn_response[n + 1]**p - magn_response[n - 1]**p)
               + 0.5)
        return n_i
    else:
        # Multi-dimensional array: recursion
        return np.array([qips(_magn_response, _n, p) for _magn_response, _n in zip(magn_response, n)])


def qi(magn_response, n):
    """Quadratic interpolation (QI) in radar echo."""
    return qips(magn_response, n, 1)


def linear_phase_interpolation(phase_response, n_i):
    """Linear interpolation of phase in radar echo."""
    if np.ndim(phase_response) == 1:
        # Single-dimensional array
        a = floor(n_i)
        b = ceil(n_i)
        phi_i = np.interp(n_i, [a, b], np.unwrap([phase_response[a], phase_response[b]]))
        return phi_i
    else:
        # Multi-dimensional array: recursion
        return np.array([linear_phase_interpolation(_phase_response, _n_i)
                         for _phase_response, _n_i in zip(phase_response, n_i)])


def compute_qips_parameter(window, fft_length=8192, n_length=1000):
    """Compute QIPS tuning parameter p to a given window function via least squares."""
    # Definition of function for optimize
    def func(n, p):
        if np.size(n) > 1:
            return np.array([func(_n, p) for _n in n])
        else:
            s = np.exp(1j * 2 * np.pi * n / fft_length * np.arange(fft_length))  # n / fft_length: normalized frequency
            s = s * getattr(windows, window)(fft_length)
            s_ft = np.abs(np.fft.fftshift(np.fft.fft(s)))
            n_est = qips(s_ft, np.argmax(s_ft), p) - fft_length / 2
            return n_est

    # Optimization (curve fit)
    n_vals = np.linspace(0, 1, n_length)
    (p_est,), _ = curve_fit(func, n_vals, n_vals, bounds=(1E-5, 1-1E-5))  # 0 < p < 1

    return p_est
