"""
signal_tool — reusable framework for backtesting leading-indicator hypotheses.

Apply a 4-section analysis (data collection → processing → pre-analysis viz →
correlation) to any pair of time series. Framework-agnostic.

Quick start:
    from signal_tool import run_pair
    run_pair("copper_transformer")
"""

from .core import Signal, SignalPair, lag_correlation, peak_correlation, compute_derivatives, align
from .sources import load_fred_monthly, load_annual
from .pairs import PAIRS
from .visualize import plot_pre_analysis, plot_correlation
from .reports import write_report

from pathlib import Path

OUTPUT_DIR = Path(__file__).parent / "output"


def run_pair(pair_name: str, output_dir: Path = None) -> dict:
    """Run the full 4-section analysis on one pair. Returns analysis dict."""
    if pair_name not in PAIRS:
        raise KeyError(f"Unknown pair '{pair_name}'. Known pairs: {list(PAIRS.keys())}")

    pair = PAIRS[pair_name]
    output_dir = output_dir or OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    # Section 1+2: Load and align
    x = pair.leading.load()
    y = pair.outcome.load()
    df = align(x, y, pair.frequency)
    df = compute_derivatives(df, pair.frequency)

    # Section 3: Pre-analysis visualization
    pre_chart = output_dir / f"{pair_name}_pre.png"
    plot_pre_analysis(df, pair, pre_chart)

    # Section 4: Correlation
    df_yoy = df.dropna(subset=["x_d1", "y_d1"])
    lag_results = lag_correlation(df_yoy["x_d1"], df_yoy["y_d1"], pair.max_lag)
    corr_chart = output_dir / f"{pair_name}_corr.png"
    plot_correlation(lag_results, pair, corr_chart)

    # Generate markdown report
    report_path = output_dir / f"{pair_name}_report.md"
    write_report(pair, df, lag_results, pre_chart, corr_chart, report_path)

    return {
        "pair": pair_name,
        "n_obs": len(df),
        "peak": peak_correlation(lag_results),
        "report": report_path,
        "pre_chart": pre_chart,
        "corr_chart": corr_chart,
    }


def run_all(output_dir: Path = None) -> list:
    """Run all defined pairs. Returns list of result dicts."""
    return [run_pair(name, output_dir) for name in PAIRS]
