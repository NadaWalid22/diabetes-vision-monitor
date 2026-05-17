from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import Rectangle


SAFE_LOW = 70
SAFE_HIGH = 180
HYPO_CRITICAL = 54
HYPER_CRITICAL = 250


def plot_glucose_trace(
    timestamps: pd.DatetimeIndex | None,
    glucose: np.ndarray,
    predictions: dict[str, np.ndarray] | None = None,
    meal_events: list[pd.Timestamp] | None = None,
    title: str = "Glucose Trace",
    save_path: str | None = None,
) -> plt.Figure:
    """
    Plot a CGM trace with safe-range bands and optional multi-model predictions.

    Args:
        timestamps:   DatetimeIndex aligned with glucose; uses integer index if None.
        glucose:      Reference CGM values (mg/dL).
        predictions:  Dict of {model_name: predictions_array} for overlay.
        meal_events:  List of timestamps where meals occurred (vertical markers).
    """
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.set_facecolor("#fafafa")

    x = timestamps if timestamps is not None else np.arange(len(glucose))

    # Safe-range shading
    ax.axhspan(SAFE_LOW, SAFE_HIGH, alpha=0.08, color="#2ecc71", label="Target range (70–180)")
    ax.axhline(SAFE_LOW, color="#e74c3c", linewidth=0.8, linestyle="--", alpha=0.7)
    ax.axhline(SAFE_HIGH, color="#e67e22", linewidth=0.8, linestyle="--", alpha=0.7)
    ax.axhline(HYPO_CRITICAL, color="#c0392b", linewidth=0.8, linestyle=":", alpha=0.5)

    # CGM trace
    ax.plot(x, glucose, color="#2c3e50", linewidth=1.5, label="CGM (reference)", zorder=3)

    # Model predictions
    colors = ["#3498db", "#9b59b6", "#e67e22", "#1abc9c"]
    if predictions:
        for (name, pred), color in zip(predictions.items(), colors):
            ax.plot(x, pred, color=color, linewidth=1.2, linestyle="--", alpha=0.85, label=name)

    # Meal markers
    if meal_events:
        for ts in meal_events:
            ax.axvline(ts, color="#f39c12", linewidth=1.0, linestyle="-", alpha=0.6)
        ax.plot([], [], color="#f39c12", linewidth=1.0, linestyle="-", alpha=0.6, label="Meal event")

    if timestamps is not None and hasattr(x, "dtype") and np.issubdtype(x.dtype, np.datetime64):
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
        ax.xaxis.set_major_locator(mdates.HourLocator(interval=2))
        fig.autofmt_xdate()

    ax.set_xlabel("Time", fontsize=11)
    ax.set_ylabel("Glucose (mg/dL)", fontsize=11)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.legend(loc="upper right", fontsize=9, framealpha=0.9)
    ax.set_ylim(30, 400)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_learning_curves(
    train_losses: list[float],
    val_losses: list[float],
    title: str = "Training Curves",
    save_path: str | None = None,
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(9, 4))
    epochs = np.arange(1, len(train_losses) + 1)
    ax.plot(epochs, train_losses, label="Train loss", color="#3498db", linewidth=1.5)
    ax.plot(epochs, val_losses, label="Validation loss", color="#e74c3c", linewidth=1.5)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE Loss")
    ax.set_title(title, fontweight="bold")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_ablation_table(
    results: dict[str, dict[str, float]],
    save_path: str | None = None,
) -> plt.Figure:
    """
    Render an ablation study results table as a figure.

    Args:
        results: {model_name: {metric_name: value}}
    """
    models = list(results.keys())
    metrics = list(next(iter(results.values())).keys())
    data = [[results[m].get(k, float("nan")) for k in metrics] for m in models]

    fig, ax = plt.subplots(figsize=(len(metrics) * 2.2 + 2, len(models) * 0.6 + 1.5))
    ax.axis("off")
    col_labels = [m.replace("_", " ").upper() for m in metrics]
    table = ax.table(
        cellText=[[f"{v:.2f}" for v in row] for row in data],
        rowLabels=models,
        colLabels=col_labels,
        cellLoc="center",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.6)

    # Highlight best (lowest) value per metric column in green
    for col_idx in range(len(metrics)):
        col_vals = [data[r][col_idx] for r in range(len(models))]
        best_row = int(np.nanargmin(col_vals))
        table[(best_row + 1, col_idx)].set_facecolor("#d4efdf")

    ax.set_title("Ablation Study Results", fontsize=13, fontweight="bold", pad=20)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig
