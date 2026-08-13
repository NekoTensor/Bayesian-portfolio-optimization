"""Walk-forward (rolling-origin) evaluation.

The rule enforced here is the one the first version of this project violated:
the weights held over period ``t`` are a function of returns strictly before
``t``. Concatenating the held-out periods gives a return series no estimator and
no optimiser ever saw, so a Sharpe ratio computed on it estimates something
forward-looking rather than restating how well the optimiser fit its own data.

Two details that are easy to get wrong and are handled explicitly:

*Weight drift.* Between rebalances the portfolio is held, not continuously
reset. Weights therefore drift with relative performance, and turnover at the
next rebalance is measured against the *drifted* holdings. Re-imposing target
weights every period would silently overstate both turnover and rebalancing
alpha.

*Cost timing.* Trading costs are charged once, to the period in which the trade
happens, rather than amortised -- so a strategy cannot escape them by rebalancing
more often.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import pandas as pd

from .estimators import Belief
from .objectives import FeasibleSet, turnover

__all__ = ["BacktestResult", "Strategy", "compare", "make_strategy", "walk_forward"]

#: ``(train_returns, prev_weights) -> target weights``
Strategy = Callable[[pd.DataFrame, "np.ndarray | None"], np.ndarray]


def make_strategy(estimator, allocator, feasible: FeasibleSet, **allocator_kwargs) -> Strategy:
    """Compose an estimator and an allocator into one walk-forward strategy.

    Keeping these separate is what lets the experiment vary one while holding the
    other fixed, which is the whole design of the study.
    """

    def strategy(train_returns: pd.DataFrame, prev_weights=None) -> np.ndarray:
        belief: Belief = estimator(train_returns)
        return allocator(belief, feasible, prev_weights, **allocator_kwargs)

    strategy.__name__ = f"{getattr(estimator, '__name__', 'est')}+{getattr(allocator, '__name__', 'alloc')}"
    return strategy


@dataclass
class BacktestResult:
    """Realised out-of-sample series and the bookkeeping behind it."""

    net_returns: pd.Series
    gross_returns: pd.Series
    weights: pd.DataFrame
    turnover: pd.Series
    frequency: int = 52
    meta: dict = field(default_factory=dict)

    @property
    def equity_curve(self) -> pd.Series:
        return (1.0 + self.net_returns).cumprod()

    def with_costs(self, cost_bps: float) -> "BacktestResult":
        """Re-charge this run at a different cost level, without refitting.

        The cost model touches only the accounting: weights are chosen by the
        strategy from training data, which never sees ``cost_bps``. So sweeping a
        cost grid by refitting is pure waste -- and worse, it invites the reader
        to assume the weights responded to costs when they did not. Charging the
        same realised turnover at a new rate is exact, not an approximation.
        """
        if cost_bps < 0:
            raise ValueError("cost_bps must be non-negative")

        net = self.gross_returns.copy()
        charges = self.turnover * cost_bps / 1e4
        net.loc[charges.index] = net.loc[charges.index] - charges

        return BacktestResult(
            net_returns=net,
            gross_returns=self.gross_returns,
            weights=self.weights,
            turnover=self.turnover,
            frequency=self.frequency,
            meta={**self.meta, "cost_bps": cost_bps,
                  "total_cost": float(self.turnover.sum() * cost_bps / 1e4)},
        )

    def metrics(self, risk_free: float = 0.0) -> dict:
        r = self.net_returns.dropna()
        if r.empty:
            return {}

        ann_return = float(r.mean()) * self.frequency
        ann_vol = float(r.std(ddof=1)) * np.sqrt(self.frequency)
        sharpe = (ann_return - risk_free) / ann_vol if ann_vol > 0 else np.nan

        curve = (1.0 + r).cumprod()
        total_return = float(curve.iloc[-1] - 1.0)
        years = len(r) / self.frequency
        cagr = float(curve.iloc[-1] ** (1.0 / years) - 1.0) if years > 0 else np.nan
        drawdown = float((curve / curve.cummax() - 1.0).min())

        downside = r[r < risk_free / self.frequency]
        downside_vol = (
            float(downside.std(ddof=1)) * np.sqrt(self.frequency) if len(downside) > 1 else np.nan
        )
        sortino = (
            (ann_return - risk_free) / downside_vol
            if downside_vol and downside_vol > 0
            else np.nan
        )

        gross_ann = float(self.gross_returns.dropna().mean()) * self.frequency

        return {
            "ann_return": ann_return,
            "cagr": cagr,
            "ann_vol": ann_vol,
            "sharpe": sharpe,
            "sortino": sortino,
            "calmar": ann_return / abs(drawdown) if drawdown < 0 else np.nan,
            "max_drawdown": drawdown,
            "total_return": total_return,
            "cost_drag_ann": gross_ann - ann_return,
            "avg_turnover": float(self.turnover.mean()) if len(self.turnover) else np.nan,
            "n_periods": int(len(r)),
            "n_rebalances": int(len(self.turnover)),
        }


def walk_forward(
    returns: pd.DataFrame,
    strategy: Strategy,
    train_window: int = 156,
    rebalance_every: int = 13,
    frequency: int = 52,
    cost_bps: float = 10.0,
    expanding: bool = False,
) -> BacktestResult:
    """Roll the origin forward, refitting on past data only.

    Parameters
    ----------
    train_window
        Observations used to form the belief. At weekly frequency 156 is three
        years -- long enough for a non-singular covariance across seven assets,
        short enough to stay responsive.
    rebalance_every
        Holding-period length in observations (13 weeks = quarterly).
    cost_bps
        Charged per unit of traded notional, so a full 200% round-trip rotation
        at 10 bps costs 20 bps of portfolio value.
    expanding
        ``True`` uses all history up to the origin instead of a fixed window.
    """
    if not isinstance(returns, pd.DataFrame):
        raise TypeError("returns must be a DataFrame of per-period simple returns")
    if train_window >= len(returns):
        raise ValueError(
            f"train_window={train_window} leaves no out-of-sample data "
            f"(panel has {len(returns)} periods)"
        )
    if rebalance_every < 1:
        raise ValueError("rebalance_every must be at least 1")

    assets = list(returns.columns)
    n = len(assets)

    net_parts: list[pd.Series] = []
    gross_parts: list[pd.Series] = []
    weight_rows: list[pd.Series] = []
    turnover_values: list[float] = []
    turnover_index: list[pd.Timestamp] = []

    held: np.ndarray | None = None  # drifted weights actually held right now

    for start in range(train_window, len(returns), rebalance_every):
        train = returns.iloc[:start] if expanding else returns.iloc[start - train_window : start]
        test = returns.iloc[start : start + rebalance_every]
        if test.empty:
            break

        target = np.asarray(strategy(train, held), dtype=float)
        if target.shape != (n,) or not np.isfinite(target).all():
            raise ValueError(f"strategy returned invalid weights at {test.index[0]}: {target}")

        traded = turnover(target, held)
        cost = traded * cost_bps / 1e4

        # Hold the target through the test window, letting weights drift.
        w = target.copy()
        gross_period: list[float] = []
        net_period: list[float] = []

        for step, (_, row) in enumerate(test.iterrows()):
            asset_returns = row.to_numpy(dtype=float)
            gross = float(w @ asset_returns)
            net_period.append(gross - (cost if step == 0 else 0.0))
            gross_period.append(gross)

            growth = 1.0 + gross
            if growth <= 0.0:
                w = np.full(n, 1.0 / n)  # total loss; nothing meaningful to drift
            else:
                w = w * (1.0 + asset_returns) / growth

        gross_parts.append(pd.Series(gross_period, index=test.index))
        net_parts.append(pd.Series(net_period, index=test.index))
        weight_rows.append(pd.Series(target, index=assets, name=test.index[0]))
        turnover_values.append(traded)
        turnover_index.append(test.index[0])

        held = w

    if not net_parts:
        raise ValueError("no out-of-sample periods were produced")

    turnover_series = pd.Series(turnover_values, index=pd.Index(turnover_index, name="Date"))

    return BacktestResult(
        net_returns=pd.concat(net_parts),
        gross_returns=pd.concat(gross_parts),
        weights=pd.DataFrame(weight_rows),
        turnover=turnover_series,
        frequency=frequency,
        meta={
            "train_window": train_window,
            "rebalance_every": rebalance_every,
            "cost_bps": cost_bps,
            "expanding": expanding,
            "total_cost": float(turnover_series.sum() * cost_bps / 1e4),
        },
    )


def compare(returns: pd.DataFrame, strategies: dict[str, Strategy], **kwargs):
    """Run several strategies over identical windows.

    Returns ``(metrics_frame, results_by_name)``; the raw results are kept so the
    significance tests in :mod:`portfolio.stats` can work on the return series
    rather than on summary statistics.
    """
    results = {name: walk_forward(returns, fn, **kwargs) for name, fn in strategies.items()}
    frame = pd.DataFrame({name: res.metrics() for name, res in results.items()}).T
    return frame.sort_values("sharpe", ascending=False), results
