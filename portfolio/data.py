"""Loading the price panel.

The cleaned weekly panel is committed to the repository rather than downloaded
on import. That is a deliberate reproducibility choice: Yahoo Finance silently
revises its adjusted-close history, so a backtest that re-downloads on every run
is not reproducible even against itself. ``experiments/fetch_data.py`` refreshes
the panel when you actually want new data.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

__all__ = ["DEFAULT_PRICES", "load_prices", "load_returns", "repo_root", "to_returns"]


def repo_root() -> Path:
    """Directory containing ``portfolio/``, so paths work from any CWD."""
    return Path(__file__).resolve().parent.parent


DEFAULT_PRICES = repo_root() / "data" / "processed" / "historical_prices_cleaned.csv"


def load_prices(path: str | Path | None = None) -> pd.DataFrame:
    """Load a wide price panel indexed by date, one column per asset."""
    path = Path(path) if path is not None else DEFAULT_PRICES
    if not path.exists():
        raise FileNotFoundError(
            f"price panel not found at {path}. Run `python experiments/fetch_data.py` "
            f"to rebuild it, or pass an explicit path."
        )

    prices = pd.read_csv(path, index_col=0, parse_dates=True).sort_index()
    prices.index.name = "Date"
    prices = prices.astype(float)

    if prices.isna().any().any():
        # Forward-fill holidays only; a leading NaN means the asset has no
        # history yet and must stay NaN so it is excluded rather than invented.
        prices = prices.ffill()
    if prices.index.has_duplicates:
        raise ValueError(f"{path} contains duplicate dates")

    return prices


def to_returns(prices: pd.DataFrame, kind: str = "simple") -> pd.DataFrame:
    """Period-over-period returns.

    ``simple`` returns aggregate correctly across assets (a portfolio return is
    the weighted mean of simple returns), which is what a backtest needs. ``log``
    is offered for distributional work, where additivity over time matters more.
    """
    if kind == "simple":
        returns = prices.pct_change(fill_method=None)
    elif kind == "log":
        returns = np.log(prices).diff()
    else:
        raise ValueError(f"kind must be 'simple' or 'log', got {kind!r}")
    return returns.dropna(how="any")


def load_returns(
    path: str | Path | None = None,
    kind: str = "simple",
    start: str | None = None,
    end: str | None = None,
) -> pd.DataFrame:
    """Convenience wrapper: load prices and difference them in one step."""
    returns = to_returns(load_prices(path), kind=kind)
    if start is not None:
        returns = returns.loc[returns.index >= pd.Timestamp(start)]
    if end is not None:
        returns = returns.loc[returns.index <= pd.Timestamp(end)]
    if returns.empty:
        raise ValueError("no observations remain after applying the date filter")
    return returns
