"""
Generate realistic synthetic CGM data for development and demo purposes.

The simulation models:
- A slow-moving physiological baseline with circadian rhythm
- Postprandial (after-meal) glucose excursions
- Nocturnal hypoglycemia events (as in GlucoSense Screen 4)
- Realistic sensor noise

Output: data/sample/synthetic_cgm.csv
         columns: datetime, glucose (mg/dL)

Usage:
    python data/sample/generate_synthetic.py
    python data/sample/generate_synthetic.py --days 14 --seed 99
"""

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd


def simulate_cgm(days: int = 7, seed: int = 42, freq_min: int = 5) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    n_points = days * 24 * 60 // freq_min
    timestamps = pd.date_range("2024-01-01", periods=n_points, freq=f"{freq_min}min")

    # --- Slow physiological baseline with circadian rhythm ---
    t = np.arange(n_points) * freq_min / 60  # hours
    baseline = 110 + 15 * np.sin(2 * np.pi * (t - 4) / 24)  # peaks mid-morning

    # --- Postprandial spikes (breakfast 8am, lunch 1pm, dinner 7pm) ---
    glucose = baseline.copy()
    meal_hours = [8, 13, 19]
    for day in range(days):
        for meal_h in meal_hours:
            meal_t = day * 24 + meal_h
            peak_magnitude = rng.uniform(40, 100)
            peak_duration = rng.uniform(1.5, 3.0)
            t_since_meal = t - meal_t
            mask = (t_since_meal >= 0) & (t_since_meal <= peak_duration * 2)
            glucose[mask] += peak_magnitude * np.exp(-((t_since_meal[mask] - peak_duration / 2) ** 2)
                                                     / (2 * (peak_duration / 3) ** 2))

    # --- Nocturnal hypoglycemia (2am, ~2x per week) ---
    for day in range(days):
        if rng.random() < 0.28:
            hypo_t = day * 24 + 2
            hypo_depth = rng.uniform(15, 35)
            hypo_duration = rng.uniform(0.5, 1.5)
            t_since_hypo = t - hypo_t
            mask = (t_since_hypo >= 0) & (t_since_hypo <= hypo_duration * 2)
            glucose[mask] -= hypo_depth * np.exp(-((t_since_hypo[mask] - hypo_duration / 2) ** 2)
                                                  / (2 * (hypo_duration / 3) ** 2))

    # --- Exercise (afternoon drop, ~every 2 days) ---
    for day in range(days):
        if rng.random() < 0.45:
            ex_t = day * 24 + rng.uniform(15, 17)
            drop = rng.uniform(20, 50)
            t_since_ex = t - ex_t
            mask = (t_since_ex >= 0) & (t_since_ex <= 2.0)
            glucose[mask] -= drop * np.exp(-t_since_ex[mask])

    # --- Sensor noise ---
    glucose += rng.normal(0, 3, n_points)

    # Clip to physiological range
    glucose = np.clip(glucose, 40, 400)

    return pd.DataFrame({"datetime": timestamps, "glucose": glucose.round(1)})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--days", type=int, default=7)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    out_dir = Path(__file__).parent
    out_path = out_dir / "synthetic_cgm.csv"

    df = simulate_cgm(days=args.days, seed=args.seed)
    df.to_csv(out_path, index=False)
    print(f"Saved {len(df)} readings ({args.days} days) → {out_path}")
    print(f"  Mean glucose: {df['glucose'].mean():.1f} mg/dL")
    print(f"  Std:          {df['glucose'].std():.1f} mg/dL")
    print(f"  Min / Max:    {df['glucose'].min():.1f} / {df['glucose'].max():.1f} mg/dL")


if __name__ == "__main__":
    main()
