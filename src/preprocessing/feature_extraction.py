from __future__ import annotations

import numpy as np


def extract_window_features(window: np.ndarray) -> np.ndarray:
    """
    Compute handcrafted statistical features from a CGM window (mg/dL).

    These can be used alongside the raw sequence or for classical ML baselines.
    Returns a 1-D feature vector of length 9.
    """
    mean = np.mean(window)
    std = np.std(window)
    slope = _linear_slope(window)
    rate_of_change = window[-1] - window[-2] if len(window) >= 2 else 0.0
    min_val = np.min(window)
    max_val = np.max(window)
    cv = std / mean if mean > 0 else 0.0                # coefficient of variation
    excursion = max_val - min_val
    last_val = window[-1]

    return np.array([mean, std, slope, rate_of_change, min_val, max_val, cv, excursion, last_val],
                    dtype=np.float32)


def extract_all_features(X: np.ndarray) -> np.ndarray:
    """
    Apply feature extraction to all windows in X.

    Args:
        X: (N, seq_len, 1) raw CGM windows
    Returns:
        (N, num_features)
    """
    windows = X[:, :, 0]   # (N, seq_len)
    return np.stack([extract_window_features(w) for w in windows])


def _linear_slope(values: np.ndarray) -> float:
    """Slope of a least-squares line fit to the window — approximates trend direction."""
    n = len(values)
    if n < 2:
        return 0.0
    t = np.arange(n, dtype=np.float64)
    t_mean = t.mean()
    v_mean = values.mean()
    num = np.sum((t - t_mean) * (values - v_mean))
    den = np.sum((t - t_mean) ** 2)
    return float(num / den) if den != 0 else 0.0
