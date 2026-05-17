from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.signal import medfilt
from sklearn.preprocessing import MinMaxScaler


GLUCOSE_LOW = 70.0    # mg/dL — clinical hypoglycemia threshold
GLUCOSE_HIGH = 180.0  # mg/dL — clinical hyperglycemia threshold
GLUCOSE_MIN = 40.0    # plausible sensor floor
GLUCOSE_MAX = 400.0   # plausible sensor ceiling


class CGMProcessor:
    """
    Cleans, normalises, and windows a raw CGM time series.

    OhioT1DM-compatible: expects a DataFrame with columns
        'datetime' (pd.Timestamp) and 'glucose' (mg/dL, float).
    Missing readings are interpolated up to a configurable gap limit.
    """

    def __init__(self, seq_len: int = 24, pred_horizons: list[int] = None):
        """
        Args:
            seq_len: Number of past readings in each input window (5-min samples).
                     24 → 2-hour context window.
            pred_horizons: Steps ahead to predict. [6, 12] → 30-min and 60-min.
        """
        self.seq_len = seq_len
        self.pred_horizons = pred_horizons or [6, 12]
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self._fitted = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def load_ohio_xml(self, xml_path: str) -> pd.DataFrame:
        """Parse an OhioT1DM XML file into a clean DataFrame."""
        import xml.etree.ElementTree as ET
        tree = ET.parse(xml_path)
        root = tree.getroot()
        records = []
        for event in root.iter("glucose_level"):
            for entry in event.iter("event"):
                ts = entry.get("ts")
                val = entry.get("value")
                if ts and val:
                    records.append({"datetime": pd.to_datetime(ts, format="%d-%m-%Y %H:%M:%S"),
                                    "glucose": float(val)})
        return pd.DataFrame(records).sort_values("datetime").reset_index(drop=True)

    def clean(self, df: pd.DataFrame, max_gap_minutes: int = 30) -> pd.DataFrame:
        """
        Remove physiologically implausible values, fill short gaps, apply
        a mild median filter to suppress sensor noise.
        """
        df = df.copy()
        df["glucose"] = df["glucose"].clip(GLUCOSE_MIN, GLUCOSE_MAX)

        # Reindex to a regular 5-min grid and interpolate short gaps
        df = df.set_index("datetime")
        df = df.reindex(pd.date_range(df.index[0], df.index[-1], freq="5min"))
        max_gap_steps = max_gap_minutes // 5
        df["glucose"] = df["glucose"].interpolate(method="time", limit=max_gap_steps)
        df = df.dropna().reset_index().rename(columns={"index": "datetime"})

        # Median filter (kernel=3) to smooth individual sensor spikes
        df["glucose"] = medfilt(df["glucose"].values, kernel_size=3)
        return df

    def fit_scaler(self, df: pd.DataFrame) -> None:
        self.scaler.fit(df[["glucose"]])
        self._fitted = True

    def make_windows(
        self, df: pd.DataFrame, normalize: bool = True
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Slide a window across the series to produce (X, y) arrays.

        Returns:
            X: (N, seq_len, 1)  — input glucose sequences
            y: (N, len(pred_horizons))  — target values at each horizon
        """
        values = df["glucose"].values.astype(np.float32)
        if normalize:
            if not self._fitted:
                raise RuntimeError("Call fit_scaler() on training data before make_windows().")
            values = self.scaler.transform(values.reshape(-1, 1)).flatten()

        max_horizon = max(self.pred_horizons)
        X, y = [], []
        for i in range(len(values) - self.seq_len - max_horizon + 1):
            X.append(values[i : i + self.seq_len])
            y.append([values[i + self.seq_len + h - 1] for h in self.pred_horizons])

        return np.array(X)[..., np.newaxis], np.array(y)   # (N, seq_len, 1), (N, horizons)

    def inverse_transform(self, values: np.ndarray) -> np.ndarray:
        """Undo normalisation on predicted glucose values."""
        flat = values.flatten().reshape(-1, 1)
        return self.scaler.inverse_transform(
            pd.DataFrame(flat, columns=["glucose"])
        ).reshape(values.shape)

    @staticmethod
    def time_in_range(glucose: np.ndarray) -> dict[str, float]:
        """Compute TIR breakdown (%), a standard clinical metric."""
        n = len(glucose)
        return {
            "tir_normal": float(np.mean((glucose >= 70) & (glucose <= 180)) * 100),
            "tir_hypo_l1": float(np.mean((glucose >= 54) & (glucose < 70)) * 100),
            "tir_hypo_l2": float(np.mean(glucose < 54) * 100),
            "tir_hyper_l1": float(np.mean((glucose > 180) & (glucose <= 250)) * 100),
            "tir_hyper_l2": float(np.mean(glucose > 250) * 100),
        }
