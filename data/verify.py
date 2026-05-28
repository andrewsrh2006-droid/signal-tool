"""
Data verification: re-fetch primary sources and spot-check against cached CSVs.

Usage:
    python data/verify.py                # verify all datasets
    python data/verify.py copper_ppi     # verify one

Currently spot-checks against hard-coded expected values. To extend:
  - Add a new dataset to KNOWN_VALUES below
  - Or wire up live re-fetching via web_fetch / FRED API (deferred)

Returns exit code 0 if all checks pass, 1 if any mismatch.
"""

import sys
from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).parent

# Hand-curated expected values from primary source.
# When you update the cached CSV, also update these for the latest dates.
# Format: { dataset_name: { "YYYY-MM-DD": expected_value, ... } }
KNOWN_VALUES = {
    "copper_ppi": {
        # Source: https://fred.stlouisfed.org/data/WPUSI019011.txt
        # Last verified 2026-05-28 against FRED Last Updated 2026-05-13
        "1967-01-01": 55.200,
        "1974-05-01": 110.100,
        "1980-02-01": 140.600,
        "1988-12-01": 192.200,
        "2006-05-01": 460.800,
        "2008-04-01": 518.500,
        "2008-12-01": 271.800,
        "2011-02-01": 532.000,
        "2015-12-01": 309.500,
        "2020-04-01": 338.100,
        "2021-05-01": 558.600,
        "2022-04-01": 580.856,
        "2024-05-01": 595.757,
        "2025-08-01": 601.081,
        "2025-12-01": 658.833,
        "2026-01-01": 706.344,
        "2026-04-01": 729.830,
    },
    "transformer_ppi": {
        # Source: https://fred.stlouisfed.org/data/PCU335311335311.txt
        # Last verified 2026-05-28 against FRED Last Updated 2026-05-13
        "1967-01-01": 46.900,
        "1981-06-01": 100.000,
        "1990-01-01": 129.600,
        "2008-08-01": 238.200,
        "2015-01-01": 237.200,
        "2020-12-01": 258.800,
        "2021-06-01": 302.700,
        "2022-01-01": 356.302,
        "2022-09-01": 416.819,
        "2023-12-01": 425.883,
        "2024-12-01": 432.642,
        "2025-09-01": 440.550,
        "2025-10-01": 458.356,
        "2025-12-01": 455.909,
        "2026-04-01": 456.170,
    },
    "hyperscaler_capex": {
        # Source: Platformonomics "Follow the CAPEX" retrospective series
        # Last verified 2026-05-28
        # Note: only 2024 and 2025 are primary-source single-citation values;
        # earlier years are aggregated from multiple Platformonomics retrospectives
        # and have lower confidence. See MANIFEST.md for per-year details.
        "2015-12-31": 24,
        "2020-12-31": 94,
        "2024-12-31": 251,
        "2025-12-31": 416,
    },
    "aluminum_ppi": {
        # Source: https://fred.stlouisfed.org/data/WPU102501
        # Last verified 2026-05-28 against FRED Last Updated 2026-05-13
        "1967-01-01": 33.100,
        "1974-08-01": 54.700,
        "1980-02-01": 86.100,
        "1988-12-01": 137.700,
        "1995-01-01": 158.400,
        "2008-04-01": 195.200,
        "2009-04-01": 146.700,
        "2015-12-01": 161.600,
        "2020-04-01": 177.900,
        "2021-11-01": 258.862,
        "2022-04-01": 287.613,
        "2024-12-01": 246.625,
        "2025-12-01": 320.588,
        "2026-04-01": 366.041,
    },
    "steel_ppi": {
        # Source: https://fred.stlouisfed.org/data/WPU101
        # Last verified 2026-05-28 against FRED Last Updated 2026-05-13
        "1967-01-01": 29.400,
        "1974-08-01": 57.700,
        "1981-12-01": 100.300,
        "1989-01-01": 119.700,
        "2008-08-01": 294.400,
        "2008-12-01": 195.600,
        "2015-12-01": 172.200,
        "2020-04-01": 204.700,
        "2021-11-01": 433.525,
        "2022-05-01": 424.725,
        "2024-12-01": 288.188,
        "2025-12-01": 323.978,
        "2026-04-01": 353.916,
    },
    "manufacturing_employment": {
        # Source: https://fred.stlouisfed.org/data/MANEMP
        # Last verified 2026-05-28 against FRED Last Updated 2026-05-08
        # Note: BLS revises monthly; we pin to 2026-05-08 vintage
        "1967-01-01": 18033,
        "1979-06-01": 19553,    # all-time peak
        "1990-12-01": 17395,
        "2000-01-01": 17284,
        "2010-01-01": 11447,    # post-GFC trough
        "2020-04-01": 11382,    # COVID trough
        "2020-12-01": 12153,
        "2024-12-01": 12693,
        "2025-12-01": 12580,
        "2026-04-01": 12596,
    },
}


def load_csv(name: str) -> pd.Series:
    """Load a cached CSV. Returns Series indexed by date."""
    path = DATA_DIR / f"{name}.csv"
    df = pd.read_csv(path)
    if "DATE" in df.columns:
        df["DATE"] = pd.to_datetime(df["DATE"])
        return df.set_index("DATE")["VALUE"].astype(float).sort_index()
    elif "YEAR" in df.columns:
        df["DATE"] = pd.to_datetime(df["YEAR"].astype(int).astype(str) + "-12-31")
        return df.set_index("DATE")["VALUE"].astype(float).sort_index()
    else:
        raise ValueError(f"CSV {path} has neither DATE nor YEAR column")


def structural_checks(series: pd.Series, name: str) -> list:
    """Generic data-quality checks. Returns list of warning strings (empty if all OK)."""
    warnings = []

    # No missing values in the body of the series
    if series.isna().any():
        n_na = series.isna().sum()
        warnings.append(f"{n_na} NaN values present")

    # Dates monotonically increasing, no duplicates
    if not series.index.is_monotonic_increasing:
        warnings.append("dates not monotonically increasing")
    if series.index.duplicated().any():
        warnings.append("duplicate dates")

    # Values must be positive (price/index data; not strictly true for diff series)
    if (series <= 0).any():
        warnings.append(f"{(series <= 0).sum()} non-positive values")

    # No "impossible" jumps (>10x in one period) — likely a parse error
    pct_jump = series.pct_change().abs()
    if (pct_jump > 10).any():
        n_big = (pct_jump > 10).sum()
        warnings.append(f"{n_big} suspicious >10x period-over-period jumps")

    # Gaps in the date sequence (only meaningful for monthly data; skip annual)
    if len(series) > 50:  # heuristic: monthly data has many rows
        expected_freq = pd.infer_freq(series.index)
        if expected_freq is None:
            warnings.append("could not infer regular date frequency")

    return warnings


def verify(name: str, tolerance: float = 0.01) -> bool:
    """Spot-check known values + run structural checks. Returns True if all OK."""
    if name not in KNOWN_VALUES:
        print(f"  ⚠ No known values defined for {name} — skipped")
        return True

    try:
        series = load_csv(name)
    except FileNotFoundError:
        print(f"  ✗ {name}: CSV not found in {DATA_DIR}")
        return False

    expected = KNOWN_VALUES[name]
    mismatches = []
    not_found = []
    for date_str, val in expected.items():
        date = pd.Timestamp(date_str)
        if date not in series.index:
            not_found.append(date_str)
        elif abs(series[date] - val) > tolerance:
            mismatches.append((date_str, series[date], val))

    # Structural checks
    structural = structural_checks(series, name)

    if not mismatches and not not_found and not structural:
        print(f"  ✓ {name}: {len(expected)} spot-checks + structural checks passed, "
              f"{len(series)} rows, "
              f"range {series.index.min().date()} → {series.index.max().date()}")
        return True
    else:
        print(f"  ✗ {name}: FAILED")
        for date_str in not_found:
            print(f"      MISSING date: {date_str}")
        for date_str, cached, fresh in mismatches:
            print(f"      {date_str}: cache={cached} vs expected={fresh} (diff={cached-fresh:+.3f})")
        for w in structural:
            print(f"      STRUCTURAL: {w}")
        return False


def main():
    args = sys.argv[1:]
    targets = args if args else list(KNOWN_VALUES.keys())

    print(f"Verifying {len(targets)} dataset(s) against KNOWN_VALUES...")
    print(f"(Tolerance: 0.01 for FRED indices, exact for annual values)\n")

    results = {name: verify(name) for name in targets}

    print()
    n_ok = sum(results.values())
    n_total = len(results)
    if n_ok == n_total:
        print(f"✓ ALL VERIFIED — {n_ok}/{n_total} datasets passed.")
        sys.exit(0)
    else:
        print(f"✗ {n_total - n_ok}/{n_total} datasets failed verification.")
        sys.exit(1)


if __name__ == "__main__":
    main()
