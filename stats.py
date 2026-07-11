"""
Statistical significance for lead-lag correlations.

Two questions a bare correlation number can't answer on its own:

  1. Could a peak this strong happen by pure luck, given that we searched many
     lags? -> permutation_pvalue()
  2. How uncertain is the correlation itself? -> block_bootstrap_ci()

Both are built to respect two facts about this data that ordinary textbook
formulas ignore:

  * The months are NOT independent — each month looks a lot like the one before
    (autocorrelation). Standard p-values assume independence and would wildly
    overstate significance here.
  * We scan many lags (-max_lag..+max_lag) and keep the best. Testing many lags
    and reporting the winner inflates the apparent strength (multiple comparisons).

The permutation test handles both: it preserves each series' own wiggle
structure, breaks only the relationship between them, and — crucially — takes
the best lag in every shuffled copy too, so the "search many lags" advantage is
baked into the null distribution.
"""

import numpy as np


def _peak_abs_r(x: np.ndarray, y: np.ndarray, max_lag: int) -> float:
    """Largest |correlation| between x and y over lags -max_lag..+max_lag."""
    best = 0.0
    n = len(x)
    for k in range(-max_lag, max_lag + 1):
        if k >= 0:
            a, b = x[: n - k], y[k:]
        else:
            a, b = x[-k:], y[: n + k]
        if len(a) < 4:
            continue
        # guard against zero-variance slices
        if np.std(a) == 0 or np.std(b) == 0:
            continue
        r = np.corrcoef(a, b)[0, 1]
        if abs(r) > abs(best):
            best = r
    return best


def permutation_pvalue(x, y, max_lag: int, n_perm: int = 1000, seed: int = 0):
    """Empirical p-value for the peak lead-lag correlation.

    Method: circular-shift surrogates. We rotate y by a random offset (wrapping
    around), which keeps y's own autocorrelation intact but destroys any true
    timing relationship with x. For each surrogate we recompute the PEAK |r|
    over all lags. The p-value is how often a surrogate's peak is at least as
    large as the real one.

    Returns (p_value, observed_peak_r, null_95th_percentile).
    """
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    n = len(x)
    observed = _peak_abs_r(x, y, max_lag)
    obs_abs = abs(observed)

    rng = np.random.default_rng(seed)
    count = 0
    null_peaks = np.empty(n_perm)
    # keep shifts well away from 0 and n so the series is genuinely de-aligned
    lo, hi = max_lag + 1, n - max_lag - 1
    for i in range(n_perm):
        shift = rng.integers(lo, hi)
        y_surr = np.roll(y, shift)
        peak = abs(_peak_abs_r(x, y_surr, max_lag))
        null_peaks[i] = peak
        if peak >= obs_abs:
            count += 1
    p = (count + 1) / (n_perm + 1)          # +1 = the observed case itself
    return p, observed, float(np.percentile(null_peaks, 95))


def block_bootstrap_ci(x, y, lag: int, n_boot: int = 1000,
                       block: int = 12, ci: float = 95, seed: int = 0):
    """Confidence interval for the correlation at a FIXED lag.

    Uses a moving-block bootstrap: instead of resampling single months (which
    would destroy the autocorrelation), we resample contiguous blocks of months
    and stitch them together, so each resample keeps realistic local structure.

    Returns (r_point, lo, hi).
    """
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    # align at the chosen lag
    if lag >= 0:
        a, b = x[: len(x) - lag], y[lag:]
    else:
        a, b = x[-lag:], y[: len(y) + lag]
    m = min(len(a), len(b))
    a, b = a[:m], b[:m]
    r_point = np.corrcoef(a, b)[0, 1] if m >= 4 and np.std(a) and np.std(b) else np.nan

    if m < block * 2:
        return r_point, np.nan, np.nan

    rng = np.random.default_rng(seed)
    n_blocks = int(np.ceil(m / block))
    starts_max = m - block
    rs = np.empty(n_boot)
    for i in range(n_boot):
        starts = rng.integers(0, starts_max + 1, n_blocks)
        idx = np.concatenate([np.arange(s, s + block) for s in starts])[:m]
        aa, bb = a[idx], b[idx]
        rs[i] = np.corrcoef(aa, bb)[0, 1] if np.std(aa) and np.std(bb) else np.nan
    lo = np.nanpercentile(rs, (100 - ci) / 2)
    hi = np.nanpercentile(rs, 100 - (100 - ci) / 2)
    return r_point, float(lo), float(hi)
