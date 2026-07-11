"""
Correlation-robustness screen (a.k.a. the false-correlation screen).

A strong correlation can still be spurious — a fluke of the data. This screen
tests whether a correlation is genuine and usable as a leading signal, rather
than an accident of noise, a single lucky period, or a coincidence.

Before any pair is called CONFIRMED, it must survive four common-sense checks.
Each check attacks a different way a correlation can be fake luck rather than a
real relationship. A pair that passes all four is much harder to dismiss as
coincidence; a pair that trips even one gets a warning.

The four checks (plain-language names in the report):

  1. SECOND OPINION  (corroboration)
     Is there at least one OTHER, independent signal pointing the same way at a
     similar lead time? One series can line up by luck. Several independent
     series lining up the same way at once is very unlikely to be luck.
     e.g. copper, aluminium and steel all leading transformer PPI ~4-8 months.

  2. SURVIVES SMOOTHING  (noise robustness)
     If we smooth out the day-to-day / month-to-month jitter, is the pattern
     still there? A real relationship survives smoothing. A pattern that only
     shows up in the raw jumpy data and vanishes when smoothed was probably
     noise pretending to be a signal.

  3. HOLDS OVER TIME  (sub-period stability)
     Split the history into an early half and a late half. Does the relationship
     show up in BOTH halves, with the same sign? A real relationship keeps
     recurring. A fake one usually only appears in one lucky stretch.

  4. HAS A REASON  (mechanism gate)
     Is there a written, common-sense reason the two things should be connected?
     Two numbers can move together for no reason at all. Requiring a stated
     mechanism stops us trusting a coincidence just because the number is high.
     NOTE: this gate only checks that a reason is WRITTEN DOWN. A human still has
     to judge whether the reason is actually sensible — the tool can't do that.

Overall verdict:
  Checks are of two kinds. GATE checks are disqualifying: "Survives smoothing"
  and "Holds over time". SUPPORT checks add confidence but never flag a signal
  on their own: "Second opinion" and "Has a reason". A lone signal is marked
  "not yet corroborated" (not failed), because with a small signal library the
  absence of a second opinion may just mean we haven't collected the signal that
  would agree yet.
  A pair PASSES only if it actually leads (peak lag >= 0), the correlation is at
  least MIN_R, and no GATE check fails. Otherwise it is FLAGGED.
"""

from dataclasses import dataclass, field
from typing import Optional
import numpy as np
import pandas as pd

from core import (
    Signal, SignalPair, align, compute_derivatives,
    lag_correlation, peak_correlation,
)


# =================================================================
# small helpers
# =================================================================

def _corr_at_lag(x: pd.Series, y: pd.Series, k: int) -> tuple:
    """corr(x[t], y[t+k]). Positive k = x leads y by k periods. Returns (r, n)."""
    if k >= 0:
        pair = pd.concat([x, y.shift(-k)], axis=1).dropna()
    else:
        pair = pd.concat([x.shift(k), y], axis=1).dropna()
    if len(pair) < 4:
        return (np.nan, len(pair))
    return (pair.iloc[:, 0].corr(pair.iloc[:, 1]), len(pair))


def _prep_yoy(pair: SignalPair) -> pd.DataFrame:
    """Load, align, take YoY of both series, drop NaN. Returns df with x_d1, y_d1."""
    x = pair.leading.load()
    y = pair.outcome.load()
    df = align(x, y, pair.frequency)
    df = compute_derivatives(df, pair.frequency)
    return df.dropna(subset=["x_d1", "y_d1"])


def _peak(pair: SignalPair, df_yoy: pd.DataFrame) -> tuple:
    """Peak lag correlation on the YoY series. Returns (lag, r, n)."""
    res = lag_correlation(df_yoy["x_d1"], df_yoy["y_d1"], pair.max_lag)
    return peak_correlation(res)


# =================================================================
# result container
# =================================================================

@dataclass
class CheckResult:
    name: str            # plain-language check name
    passed: Optional[bool]   # True / False / None (= not applicable / not yet testable)
    detail: str          # one-line human explanation
    kind: str = "gate"   # "gate" = disqualifying · "support" = confirmatory only


@dataclass
class ScreenResult:
    pair_name: str
    peak_lag: float
    peak_r: float
    n: int
    leads: bool                      # peak_lag >= 0
    checks: list = field(default_factory=list)
    overall_pass: bool = False

    @property
    def failed_checks(self) -> list:
        return [c.name for c in self.checks if c.passed is False]


# =================================================================
# the four checks
# =================================================================

# thresholds (kept deliberately simple and explicit)
MIN_R = 0.30                 # a correlation weaker than this we don't treat as a signal
LAG_TOLERANCE_MONTHLY = 6    # months a corroborating lag may differ and still "agree"
LAG_TOLERANCE_ANNUAL = 1     # years
SMOOTH_WINDOW_MONTHLY = 6    # months of rolling-mean smoothing
SUBPERIOD_MIN_R = 0.20       # each half must reach at least this |r| at the peak lag


def check_second_opinion(target: SignalPair, target_peak: tuple,
                         all_pairs: dict, all_peaks: dict) -> CheckResult:
    """PASS if >=1 OTHER pair with the SAME outcome agrees in sign and similar lag."""
    t_lag, t_r, _ = target_peak
    tol = LAG_TOLERANCE_MONTHLY if target.frequency == "monthly" else LAG_TOLERANCE_ANNUAL
    corroborators = []
    for name, other in all_pairs.items():
        if name == target.name:
            continue
        if other.outcome.name != target.outcome.name:
            continue                       # must predict the same thing
        o_lag, o_r, o_n = all_peaks[name]
        if np.isnan(o_r):
            continue
        same_sign = np.sign(o_r) == np.sign(t_r)
        similar_lag = abs(o_lag - t_lag) <= tol
        strong_enough = abs(o_r) >= MIN_R
        both_lead = (o_lag >= 0) and (t_lag >= 0)
        if same_sign and similar_lag and strong_enough and both_lead:
            corroborators.append(f"{other.leading.name} (r={o_r:+.2f} @ {o_lag:+d})")
    if corroborators:
        return CheckResult("Second opinion", True,
                           "backed up by " + "; ".join(corroborators), kind="support")
    return CheckResult("Second opinion", None,
                       "not yet corroborated — no other signal in the library agrees at a "
                       "similar lead (this becomes meaningful as more signals are added)",
                       kind="support")


def check_survives_smoothing(pair: SignalPair, df_yoy: pd.DataFrame,
                             raw_peak: tuple) -> CheckResult:
    """PASS if, after smoothing, sign holds, |r| stays >= MIN_R, lag stays close."""
    if pair.frequency != "monthly" or len(df_yoy) < SMOOTH_WINDOW_MONTHLY * 3:
        return CheckResult("Survives smoothing", None,
                           "too few / too coarse data points to smooth meaningfully")
    w = SMOOTH_WINDOW_MONTHLY
    xs = df_yoy["x_d1"].rolling(w, center=True).mean()
    ys = df_yoy["y_d1"].rolling(w, center=True).mean()
    sm = pd.concat([xs.rename("x"), ys.rename("y")], axis=1).dropna()
    res = lag_correlation(sm["x"], sm["y"], pair.max_lag)
    s_lag, s_r, s_n = peak_correlation(res)
    r_lag, r_r, _ = raw_peak
    same_sign = np.sign(s_r) == np.sign(r_r)
    close_lag = abs(s_lag - r_lag) <= LAG_TOLERANCE_MONTHLY
    strong = abs(s_r) >= MIN_R
    if same_sign and close_lag and strong:
        return CheckResult("Survives smoothing", True,
                           f"smoothed peak r={s_r:+.2f} @ {s_lag:+d} mo — pattern persists")
    return CheckResult("Survives smoothing", False,
                       f"smoothed peak r={s_r:+.2f} @ {s_lag:+d} mo — weakens/shifts when de-noised")


def check_holds_over_time(pair: SignalPair, df_yoy: pd.DataFrame,
                          raw_peak: tuple) -> CheckResult:
    """PASS if the relationship at the peak lag shows the same sign in BOTH halves."""
    lag, r, _ = raw_peak
    n = len(df_yoy)
    if n < 8:
        return CheckResult("Holds over time", None,
                           "too few data points to split into two halves")
    mid = n // 2
    first = df_yoy.iloc[:mid]
    second = df_yoy.iloc[mid:]
    r1, n1 = _corr_at_lag(first["x_d1"], first["y_d1"], lag)
    r2, n2 = _corr_at_lag(second["x_d1"], second["y_d1"], lag)
    if np.isnan(r1) or np.isnan(r2):
        return CheckResult("Holds over time", None,
                           "not enough overlap in one half to measure")
    same_sign = np.sign(r1) == np.sign(r2) == np.sign(r)
    both_present = abs(r1) >= SUBPERIOD_MIN_R and abs(r2) >= SUBPERIOD_MIN_R
    early = first.index[0].year
    split = second.index[0].year
    if same_sign and both_present:
        return CheckResult("Holds over time", True,
                           f"early ({early}+) r={r1:+.2f}, late ({split}+) r={r2:+.2f} — recurs in both eras")
    return CheckResult("Holds over time", False,
                       f"early ({early}+) r={r1:+.2f}, late ({split}+) r={r2:+.2f} — not consistent across eras")


def check_has_reason(pair: SignalPair) -> CheckResult:
    """PASS if a mechanism/hypothesis is written down. Human must still judge quality."""
    text = (pair.hypothesis or "").strip()
    if len(text) >= 20:
        return CheckResult("Has a reason", True,
                           f'stated: "{text[:90]}..." (human should confirm this is sensible)',
                           kind="support")
    return CheckResult("Has a reason", None,
                       "no written mechanism yet — why should these two move together?",
                       kind="support")


# =================================================================
# top-level screen
# =================================================================

def screen_all(all_pairs: dict) -> dict:
    """Run every pair through all four checks. Returns {pair_name: ScreenResult}."""
    # First pass: compute each pair's peak once (needed for the 'second opinion' check).
    dfs, peaks = {}, {}
    for name, pair in all_pairs.items():
        df = _prep_yoy(pair)
        dfs[name] = df
        peaks[name] = _peak(pair, df)

    results = {}
    for name, pair in all_pairs.items():
        df = dfs[name]
        pk = peaks[name]
        lag, r, n = pk
        leads = (not np.isnan(r)) and lag >= 0

        checks = [
            check_second_opinion(pair, pk, all_pairs, peaks),
            check_survives_smoothing(pair, df, pk),
            check_holds_over_time(pair, df, pk),
            check_has_reason(pair),
        ]

        # Only GATE checks are disqualifying. SUPPORT checks (second opinion,
        # has a reason) add confidence but never flag a signal on their own — this
        # stops a lone-but-real signal from being failed just because the signal
        # library is still small.
        gate_ok = all(c.passed is not False for c in checks if c.kind == "gate")
        overall = leads and gate_ok and abs(r) >= MIN_R

        results[name] = ScreenResult(
            pair_name=name, peak_lag=lag, peak_r=r, n=n,
            leads=leads, checks=checks, overall_pass=overall,
        )
    return results
