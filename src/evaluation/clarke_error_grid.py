"""
Clarke Error Grid Analysis (EGA) for glucose prediction evaluation.

The EGA divides predicted vs. reference glucose into five clinical zones:
  A — Clinically accurate (within ±20% or both in normal range)
  B — Benign errors (outside 20% but no dangerous treatment decision)
  C — Overcorrection errors (treatment would cause dangerous correction)
  D — Dangerous failure to detect (missed hypo/hyperglycemia)
  E — Erroneous treatment (treatment is opposite to what is needed)

Reference: Clarke et al., Diabetes Care 1987; 10(5):622-628.
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


def zone(ref: float, pred: float) -> str:
    """Classify a single (reference, prediction) pair into a Clarke EGA zone."""
    # Zone A
    if (ref < 70 and pred < 70) or (abs(pred - ref) / max(ref, 1e-6) <= 0.20):
        return "A"

    # Zone E
    if (ref >= 180 and pred <= 70) or (ref <= 70 and pred >= 180):
        return "E"

    # Zone D
    if (ref >= 240 and 70 <= pred <= 180) or (ref <= 54 and pred >= 70):
        return "D"

    # Zone C
    if (ref > 70 and pred > ref + 110) or (ref < 180 and pred < ref - 110):
        return "C"

    return "B"


def clarke_error_grid(
    y_true: np.ndarray, y_pred: np.ndarray
) -> tuple[dict[str, float], dict[str, np.ndarray]]:
    """
    Compute Clarke EGA zone percentages for a set of predictions.

    Args:
        y_true: Reference glucose values (mg/dL), shape (N,)
        y_pred: Predicted glucose values (mg/dL), shape (N,)

    Returns:
        percentages: dict with keys 'A'–'E' giving % of points in each zone
        indices:     dict with the same keys giving boolean index arrays
    """
    y_true = np.asarray(y_true, dtype=np.float64).flatten()
    y_pred = np.asarray(y_pred, dtype=np.float64).flatten()
    assert len(y_true) == len(y_pred), "Arrays must have the same length."

    zones = np.array([zone(r, p) for r, p in zip(y_true, y_pred)])
    n = len(zones)

    percentages = {z: float(np.sum(zones == z) / n * 100) for z in "ABCDE"}
    indices = {z: zones == z for z in "ABCDE"}
    return percentages, indices


def plot_clarke_error_grid(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    title: str = "Clarke Error Grid",
    save_path: str | None = None,
) -> plt.Figure:
    """
    Render the Clarke Error Grid plot with zone boundary lines.

    Points in zone A (safe) are plotted in green; E (dangerous) in red.
    """
    y_true = np.asarray(y_true, dtype=np.float64).flatten()
    y_pred = np.asarray(y_pred, dtype=np.float64).flatten()
    percentages, indices = clarke_error_grid(y_true, y_pred)

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.set_facecolor("#f9f9f9")

    colors = {"A": "#2ecc71", "B": "#3498db", "C": "#f39c12", "D": "#e67e22", "E": "#e74c3c"}
    for z, color in colors.items():
        mask = indices[z]
        ax.scatter(y_true[mask], y_pred[mask], s=8, alpha=0.6, color=color, label=f"Zone {z}: {percentages[z]:.1f}%")

    # --- Zone boundary lines (from Clarke 1987) ---
    ax.plot([0, 400], [0, 400], "k-", linewidth=1, alpha=0.4)          # identity line
    ax.plot([0, 175 / 3], [70, 70], "k--", linewidth=0.8, alpha=0.6)
    ax.plot([175 / 3, 400], [70, 400 * 6 / 5 - 40], "k--", linewidth=0.8, alpha=0.6)
    ax.plot([70, 70], [84, 400], "k--", linewidth=0.8, alpha=0.6)
    ax.plot([0, 70], [180, 180], "k--", linewidth=0.8, alpha=0.6)
    ax.plot([70, 400], [180, 400], "k--", linewidth=0.8, alpha=0.6)
    ax.plot([180, 400], [70, 70], "k--", linewidth=0.8, alpha=0.6)
    ax.plot([180, 180], [0, 70], "k--", linewidth=0.8, alpha=0.6)
    ax.plot([240, 240], [70, 180], "k--", linewidth=0.8, alpha=0.6)
    ax.plot([240, 400], [180, 180], "k--", linewidth=0.8, alpha=0.6)
    ax.plot([130, 180], [0, 70], "k--", linewidth=0.8, alpha=0.6)

    # Zone labels
    for label, (tx, ty) in {
        "A": (150, 280), "B": (60, 260), "C": (310, 110), "D": (320, 40), "E": (40, 340)
    }.items():
        ax.text(tx, ty, label, fontsize=20, fontweight="bold", alpha=0.25, ha="center")

    ax.set_xlim(0, 400)
    ax.set_ylim(0, 400)
    ax.set_xlabel("Reference Glucose (mg/dL)", fontsize=12)
    ax.set_ylabel("Predicted Glucose (mg/dL)", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.legend(loc="upper left", fontsize=9, framealpha=0.9)
    ax.add_patch(mpatches.FancyBboxPatch((0, 0), 400, 400, linewidth=1,
                                          edgecolor="gray", facecolor="none",
                                          boxstyle="square,pad=0"))
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig
