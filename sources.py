"""
Data source loaders. Each function returns a pd.Series indexed by date.

Add a new source by writing a function here and referencing it in pairs.py.

Source naming convention:
  - Monthly time series: index is DatetimeIndex with monthly frequency
  - Annual time series:  index is DatetimeIndex with year-end frequency

All data is loaded from cached CSV files in ./data/ so the tool runs
without network access. To refresh data, use scripts/fetch_*.py
(not yet implemented — currently CSVs are populated manually via web_fetch).
"""

from pathlib import Path
import pandas as pd

DATA_DIR = Path(__file__).parent / "data"


def load_fred_monthly(series_id: str) -> pd.Series:
    """Load a FRED monthly time series from data/<series_id>.csv.

    Expected CSV format: two columns DATE, VALUE (DATE is YYYY-MM-DD).
    """
    path = DATA_DIR / f"{series_id}.csv"
    df = pd.read_csv(path, parse_dates=["DATE"])
    return df.set_index("DATE")["VALUE"].astype(float).sort_index()


def load_annual(filename: str) -> pd.Series:
    """Load an annual time series from data/<filename>.csv.

    Expected CSV format: two columns YEAR, VALUE.
    Returns a Series indexed by year-end Timestamp.
    """
    path = DATA_DIR / filename
    df = pd.read_csv(path)
    df["DATE"] = pd.to_datetime(df["YEAR"].astype(int).astype(str) + "-12-31")
    return df.set_index("DATE")["VALUE"].astype(float).sort_index()
