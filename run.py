#!/usr/bin/env python3
"""
CLI entry point for signal_tool.

Usage:
    python run.py                          # run all pairs
    python run.py copper_transformer       # run one pair
    python run.py --list                   # list available pairs
"""

import sys
from pathlib import Path

# Add parent to path so we can import the package
sys.path.insert(0, str(Path(__file__).parent.parent))
from signal_tool import run_pair, run_all, PAIRS


def main():
    args = sys.argv[1:]

    if not args or args[0] in ("--all", "-a"):
        print(f"Running all {len(PAIRS)} pairs...")
        results = run_all()
    elif args[0] in ("--list", "-l"):
        print("Available signal pairs:")
        for name, pair in PAIRS.items():
            print(f"  {name}")
            print(f"    Hypothesis: {pair.hypothesis}")
            print(f"    Frequency: {pair.frequency}, expected lead: {pair.expected_lead_low}-{pair.expected_lead_high} {pair.lag_unit}")
            print()
        return
    elif args[0] in ("--help", "-h"):
        print(__doc__)
        return
    else:
        results = [run_pair(args[0])]

    print("\nResults:")
    for r in results:
        lag, peak_r, n = r["peak"]
        print(f"  {r['pair']:30s}  peak r={peak_r:+.3f} at lag={lag:+d}  (n={n}, total_obs={r['n_obs']})")
        print(f"    Report: {r['report']}")


if __name__ == "__main__":
    main()
