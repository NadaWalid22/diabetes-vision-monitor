from __future__ import annotations

import numpy as np


def mard(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Mean Absolute Relative Difference — the standard CGM accuracy metric.
    Analogous to MAPE but named MARD in clinical literature.

    MARD < 10% is considered clinically acceptable for CGM sensors.
    We use the same standard to evaluate prediction models.
    """
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    mask = y_true > 0
    return float(np.mean(np.abs(y_true[mask] - y_pred[mask]) / y_true[mask]) * 100)


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(np.asarray(y_true) - np.asarray(y_pred))))


def compute_all_metrics(y_true: np.ndarray, y_pred: np.ndarray, horizon_minutes: int = 30) -> dict:
    """
    Full evaluation suite for a single prediction horizon.

    Returns a flat dict ready to log to W&B or print as a results table.
    """
    return {
        f"mard_{horizon_minutes}min": mard(y_true, y_pred),
        f"rmse_{horizon_minutes}min": rmse(y_true, y_pred),
        f"mae_{horizon_minutes}min": mae(y_true, y_pred),
    }
