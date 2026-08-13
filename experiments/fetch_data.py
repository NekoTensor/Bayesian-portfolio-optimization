"""Rebuild the weekly price panel from Yahoo Finance.

Not run as part of the analysis. The panel under ``data/`` is committed so that
results reproduce byte-for-byte offline; this script exists so the data provenance
is auditable and the window can be extended, not so every run re-downloads.

    python experiments/fetch_data.py --start 2012-01-01 --end 2023-12-31

A note on the universe. Seven names is small, and they were originally chosen
without a stated rationale -- a fair criticism. They are kept here for continuity
with the committed results, but the composition is defensible after the fact:
three mega-cap technology names with high idiosyncratic variance (MSFT, GOOGL,
AMZN), three lower-beta industrials/staples (MMM, GE, HSY), and one short-duration
Treasury ETF (SHY) as the low-volatility anchor. That spread is what makes the
covariance matrix non-trivial. ``--tickers`` accepts any universe; re-running the
study on a wider one is the most obvious extension, and the numbers in
``results/`` will change accordingly.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from portfolio.data import repo_root

DEFAULT_TICKERS = ["AMZN", "GE", "GOOGL", "HSY", "MMM", "MSFT", "SHY"]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tickers", nargs="+", default=DEFAULT_TICKERS)
    parser.add_argument("--start", default="2012-01-01")
    parser.add_argument("--end", default="2023-12-31")
    parser.add_argument("--interval", default="1wk", choices=["1d", "1wk", "1mo"])
    args = parser.parse_args()

    try:
        import yfinance as yf
    except ImportError:
        print("yfinance is not installed. Install it with:  pip install 'yfinance>=0.2.40'")
        return 1

    print(f"downloading {len(args.tickers)} tickers, {args.start} to {args.end} ...")
    raw = yf.download(
        args.tickers, start=args.start, end=args.end,
        interval=args.interval, auto_adjust=True, progress=False,
    )
    if raw.empty:
        print("download returned no rows -- check connectivity and ticker symbols")
        return 1

    prices = raw["Close"] if isinstance(raw.columns, pd.MultiIndex) else raw
    prices = prices[sorted(prices.columns)].sort_index()

    raw_dir = repo_root() / "data" / "raw"
    processed_dir = repo_root() / "data" / "processed"
    raw_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)

    prices.to_csv(raw_dir / "historical_prices.csv")

    # Drop any date where an asset has no price rather than interpolating one:
    # an invented price becomes an invented return, and the covariance inherits it.
    before = len(prices)
    cleaned = prices.dropna(how="any")
    cleaned.index.name = "Date"
    cleaned.to_csv(processed_dir / "historical_prices_cleaned.csv")

    print(f"  raw       : {before} rows -> {raw_dir / 'historical_prices.csv'}")
    print(f"  cleaned   : {len(cleaned)} rows ({before - len(cleaned)} dropped for missing data)")
    print(f"  range     : {cleaned.index.min().date()} to {cleaned.index.max().date()}")
    print(f"  written to: {processed_dir / 'historical_prices_cleaned.csv'}")
    print("\nre-run `python experiments/run_backtest.py` to refresh results/.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
