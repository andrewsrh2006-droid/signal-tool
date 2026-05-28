"""
Core data structures and analysis functions.

Signal:      a single time series with metadata (source, units, URL)
SignalPair:  two Signals + a hypothesis to test
analyze():   run the full 4-section analysis on any pair

Methodology follows feedback_backtest_methodology.md:
  Section 1 — data collection (handled by sources.py + Signal metadata)
  Section 2 — data processing (derivatives, alignment)
  Section 3 — pre-analysis visualization (3-panel: level, 1st, 2nd derivative)
  Section 4 — correlation analysis (lag correlation curve)
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Tuple, Optional
import pandas as pd
import numpy as np


# =================================================================
# DATA STRUCTURES
# =================================================================

@dataclass
class Signal:
    name: str                          # short identifier
    description: str                   # human-readable description
    source_url: str                    # where the data comes from
    units: str                         # what the values mean
    loader: Callable                   # function that returns pd.Series when called with loader_args
    loader_args: tuple = ()            # positional args for loader

    def load(self) -> pd.Series:
        return self.loader(*self.loader_args)


@dataclass
class SignalPair:
    name: str                          # short identifier (used in output filenames)
    leading: Signal                    # the signal we hypothesize as leading
    outcome: Signal                    # the variable we want to predict
    hypothesis: str                    # one-line statement of expected relationship
    expected_lead_low: float           # months (or years if annual)
    expected_lead_high: float          # months (or years if annual)
    frequency: str = "monthly"         # "monthly" or "annual"

    @property
    def lag_unit(self) -> str:
        return "months" if self.frequency == "monthly" else "years"

    @property
    def max_lag(self) -> int:
        return 24 if self.frequency == "monthly" else 5


# =================================================================
# SECTION 2 — DATA PROCESSING (derivatives, alignment)
# =================================================================

def first_derivative(s: pd.Series, freq: str) -> pd.Series:
    """YoY % change. For monthly data, uses 12-period shift. For annual, 1-period."""
    period = 12 if freq == "monthly" else 1
    return (s.pct_change(period) * 100).rename(f"{s.name}_yoy_pct")


def second_derivative(yoy: pd.Series, freq: str) -> pd.Series:
    """Change in YoY % from one period to the next."""
    period = 12 if freq == "monthly" else 1
    return yoy.diff(period).rename(yoy.name.replace("_yoy_pct", "_2nd"))


def align(x: pd.Series, y: pd.Series, freq: str) -> pd.DataFrame:
    """Concat two series on a common index. For annual, resample to year-end mean first."""
    if freq == "annual":
        x = x.resample("YE").mean()
        y = y.resample("YE").mean()
    df = pd.concat([x.rename("x"), y.rename("y")], axis=1).dropna()
    return df


def compute_derivatives(df: pd.DataFrame, freq: str) -> pd.DataFrame:
    """Add 1st and 2nd derivative columns to an aligned (x, y) frame."""
    df = df.copy()
    df["x_d1"] = first_derivative(df["x"], freq)
    df["y_d1"] = first_derivative(df["y"], freq)
    df["x_d2"] = second_derivative(df["x_d1"], freq)
    df["y_d2"] = second_derivative(df["y_d1"], freq)
    return df


# =================================================================
# SECTION 4 — CORRELATION ANALYSIS
# =================================================================

def lag_correlation(x: pd.Series, y: pd.Series, max_lag: int) -> list:
    """corr(x[t], y[t+k]) for k = -max_lag ... +max_lag.

    Positive k means x leads y by k periods. Returns list of (lag, r, n).
    """
    out = []
    for k in range(-max_lag, max_lag + 1):
        if k >= 0:
            pair = pd.concat([x, y.shift(-k)], axis=1).dropna()
        else:
            pair = pd.concat([x.shift(k), y], axis=1).dropna()
        if len(pair) < 4:
            out.append((k, np.nan, len(pair)))
        else:
            r = pair.iloc[:, 0].corr(pair.iloc[:, 1])
            out.append((k, r, len(pair)))
    return out


def peak_correlation(lag_results: list) -> Tuple[int, float, int]:
    """Find the lag with maximum absolute correlation. Returns (lag, r, n)."""
    valid = [(k, r, n) for k, r, n in lag_results if not np.isnan(r)]
    if not valid:
        return (0, np.nan, 0)
    return max(valid, key=lambda t: abs(t[1]))
