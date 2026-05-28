"""
Visualization functions. Produces two charts per pair:

1. Pre-analysis chart (3 panels): level, 1st derivative, 2nd derivative
2. Correlation chart (1 panel): lag correlation curve with peak highlighted

Both are PNG files saved to output/.
"""

from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


LEADING_COLOR = "#5B8FF9"
OUTCOME_COLOR = "#1F4E78"


def _format_x_axis(ax, df, freq):
    if freq == "annual":
        ax.set_xticks(df.index)
        ax.set_xticklabels([str(d.year) for d in df.index], rotation=0)


def plot_pre_analysis(df: pd.DataFrame, pair, out_path: Path):
    """3-panel pre-analysis chart: level, 1st derivative, 2nd derivative."""
    freq = pair.frequency
    fig, axes = plt.subplots(3, 1, figsize=(12, 12),
                              gridspec_kw={"height_ratios": [1, 1, 1]})

    # --- Panel 1: Level ---
    ax1 = axes[0]
    ax1b = ax1.twinx()
    if freq == "annual":
        x_pos = range(len(df))
        ax1.bar(x_pos, df["x"], color=LEADING_COLOR, alpha=0.85)
        ax1b.plot(x_pos, df["y"], color=OUTCOME_COLOR, linewidth=2.4,
                  marker="o", markersize=6)
        ax1.set_xticks(x_pos)
        ax1.set_xticklabels([str(d.year) for d in df.index])
    else:
        ax1.plot(df.index, df["x"], color=LEADING_COLOR, linewidth=1.4)
        ax1b.plot(df.index, df["y"], color=OUTCOME_COLOR, linewidth=1.4)

    ax1.set_ylabel(pair.leading.name, color=LEADING_COLOR)
    ax1b.set_ylabel(pair.outcome.name, color=OUTCOME_COLOR)
    ax1.set_title(f"LEVEL (0th derivative) — {pair.leading.name} vs {pair.outcome.name}",
                  fontsize=12, fontweight="bold", loc="left")
    ax1.tick_params(axis="y", labelcolor=LEADING_COLOR)
    ax1b.tick_params(axis="y", labelcolor=OUTCOME_COLOR)
    ax1.grid(True, alpha=0.3, axis="y")

    # --- Panel 2: 1st derivative ---
    ax2 = axes[1]
    df_d1 = df.dropna(subset=["x_d1", "y_d1"])
    if freq == "annual":
        x_pos = range(len(df_d1))
        width = 0.35
        ax2.bar([p - width/2 for p in x_pos], df_d1["x_d1"], width,
                color=LEADING_COLOR, label=f"{pair.leading.name} YoY %")
        ax2.bar([p + width/2 for p in x_pos], df_d1["y_d1"], width,
                color=OUTCOME_COLOR, label=f"{pair.outcome.name} YoY %")
        ax2.set_xticks(x_pos)
        ax2.set_xticklabels([str(d.year) for d in df_d1.index])
    else:
        ax2.axhline(0, color="black", linewidth=0.5)
        ax2.plot(df_d1.index, df_d1["x_d1"], color=LEADING_COLOR,
                 linewidth=1.4, alpha=0.85, label=f"{pair.leading.name} YoY %")
        ax2.plot(df_d1.index, df_d1["y_d1"], color=OUTCOME_COLOR,
                 linewidth=1.6, label=f"{pair.outcome.name} YoY %")

    ax2.axhline(0, color="black", linewidth=0.5)
    ax2.set_ylabel("YoY % change")
    ax2.set_title("1st DERIVATIVE — how fast each is moving (velocity)",
                  fontsize=12, fontweight="bold", loc="left")
    ax2.legend(loc="upper left")
    ax2.grid(True, alpha=0.3, axis="y")

    # --- Panel 3: 2nd derivative ---
    ax3 = axes[2]
    df_d2 = df.dropna(subset=["x_d2", "y_d2"])
    if freq == "annual":
        x_pos = range(len(df_d2))
        width = 0.35
        ax3.bar([p - width/2 for p in x_pos], df_d2["x_d2"], width,
                color=LEADING_COLOR, label=f"{pair.leading.name} 2nd deriv")
        ax3.bar([p + width/2 for p in x_pos], df_d2["y_d2"], width,
                color=OUTCOME_COLOR, label=f"{pair.outcome.name} 2nd deriv")
        ax3.set_xticks(x_pos)
        ax3.set_xticklabels([str(d.year) for d in df_d2.index])
    else:
        ax3.plot(df_d2.index, df_d2["x_d2"], color=LEADING_COLOR,
                 linewidth=1.2, alpha=0.7, label=f"{pair.leading.name} 2nd deriv")
        ax3.plot(df_d2.index, df_d2["y_d2"], color=OUTCOME_COLOR,
                 linewidth=1.4, label=f"{pair.outcome.name} 2nd deriv")

    ax3.axhline(0, color="black", linewidth=0.5)
    ax3.set_ylabel("Change in YoY %")
    ax3.set_xlabel(pair.lag_unit.capitalize()[:-1])  # "Year" or "Month"
    ax3.set_title("2nd DERIVATIVE — whether accelerating or decelerating",
                  fontsize=12, fontweight="bold", loc="left")
    ax3.legend(loc="upper left")
    ax3.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig(out_path, dpi=120, bbox_inches="tight", facecolor="white")
    plt.close()


def plot_correlation(lag_results: list, pair, out_path: Path):
    """Lag-correlation chart with peak highlighted."""
    lags = [r[0] for r in lag_results]
    corrs = [r[1] for r in lag_results]
    ns = [r[2] for r in lag_results]

    valid_corrs = [c for c in corrs if not np.isnan(c)]
    if not valid_corrs:
        peak = 0
    else:
        peak_idx = int(np.argmax([abs(c) if not np.isnan(c) else -1 for c in corrs]))
        peak = corrs[peak_idx]

    fig, ax = plt.subplots(figsize=(11, 5))
    colors = [OUTCOME_COLOR if c == peak else "#9BB7D4" for c in corrs]
    ax.bar(lags, corrs, color=colors)
    ax.axhline(0, color="black", linewidth=0.5)

    for lag, r, n in lag_results:
        if not np.isnan(r) and (lag == lag_results[peak_idx][0]
                                 or lag == 0
                                 or abs(r) > 0.4):
            offset = 0.03 if r >= 0 else -0.06
            ax.text(lag, r + offset, f"{r:+.2f}",
                    ha="center", fontsize=8)

    ax.set_xlabel(f"{pair.lag_unit.capitalize()} {pair.leading.name} leads {pair.outcome.name}")
    ax.set_ylabel("Correlation (r)")
    ax.set_title(f"LAG CORRELATION — {pair.name}",
                 fontsize=12, fontweight="bold", loc="left")
    ax.set_xticks(lags)
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig(out_path, dpi=120, bbox_inches="tight", facecolor="white")
    plt.close()
