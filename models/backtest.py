"""Walk-forward (rolling-origin) evaluation.

updated the method...now we have the weights for period t are chosen using only data strictly before period t.
Concatenating those held-out periods gives a return series the optimizer
never saw, so Sharpe and max-drawdown computed on it mean something.
"""
from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np
import pandas as pd
from .objectives import turnover


@dataclass
class BacktestResult:
    returns: pd.Series          # net-of-cost out-of-sample returns
    gross_returns: pd.Series
    weights: pd.DataFrame
    frequency: int = 52
    meta: dict = field(default_factory=dict)

    @property
    def equity_curve(self) -> pd.Series:
        return (1.0 + self.returns).cumprod()

    def metrics(self, risk_free: float = 0.0) -> dict:
        r = self.returns.dropna()
        if r.empty:
            return {}
        ann_return = float(r.mean()) * self.frequency
        ann_vol = float(r.std(ddof=1)) * np.sqrt(self.frequency)
        sharpe = (ann_return - risk_free) / ann_vol if ann_vol > 0 else np.nan

        curve = (1.0 + r).cumprod()
        drawdown = curve / curve.cummax() - 1.0

        downside = r[r < 0]
        downside_vol = (float(downside.std(ddof=1)) * np.sqrt(self.frequency)
                        if len(downside) > 1 else np.nan)
        sortino = ((ann_return - risk_free) / downside_vol
                   if downside_vol and downside_vol > 0 else np.nan)

        return {
            "ann_return": ann_return, "ann_vol": ann_vol,
            "sharpe": sharpe, "sortino": sortino,
            "max_drawdown": float(drawdown.min()),
            "total_return": float(curve.iloc[-1] - 1.0),
            "n_periods": int(len(r)),
            "n_rebalances": int(self.meta.get("n_rebalances", 0)),
            "avg_turnover": float(self.meta.get("avg_turnover", np.nan)),
            "total_cost": float(self.meta.get("total_cost", np.nan)),
        }


def walk_forward(returns, fit_fn, train_window=156, rebalance_every=13,
                 frequency=52, cost_bps=10.0, expanding=False) -> BacktestResult:
    if train_window >= len(returns):
        raise ValueError(f"train_window={train_window} leaves no out-of-sample data "
                         f"(only {len(returns)} periods available)")

    assets = list(returns.columns)
    net_chunks, gross_chunks, weight_rows = [], [], []
    prev_weights = None
    turnovers, total_cost = [], 0.0

    for start in range(train_window, len(returns), rebalance_every):
        train = (returns.iloc[0:start] if expanding
                 else returns.iloc[start - train_window:start])
        test = returns.iloc[start:start + rebalance_every]
        if test.empty:
            break

        w = np.asarray(fit_fn(train, prev_weights), dtype=float)
        if w.shape != (len(assets),) or not np.isfinite(w).all():
            raise ValueError(f"fit_fn returned invalid weights at index {start}: {w}")

        gross = test.values @ w
        tno = turnover(w, prev_weights if prev_weights is not None
                       else np.zeros(len(assets)))
        cost = tno * cost_bps / 1e4

        net = gross.copy()
        net[0] -= cost   # charge rebalance cost to first period of holding window

        gross_chunks.append(pd.Series(gross, index=test.index))
        net_chunks.append(pd.Series(net, index=test.index))
        weight_rows.append(pd.Series(w, index=assets, name=test.index[0]))

        turnovers.append(tno)
        total_cost += cost
        prev_weights = w

    return BacktestResult(
        returns=pd.concat(net_chunks),
        gross_returns=pd.concat(gross_chunks),
        weights=pd.DataFrame(weight_rows),
        frequency=frequency,
        meta={"n_rebalances": len(turnovers),
              "avg_turnover": float(np.mean(turnovers)) if turnovers else np.nan,
              "total_cost": total_cost, "cost_bps": cost_bps,
              "train_window": train_window, "rebalance_every": rebalance_every},
    )


def compare(returns, strategies: dict, **kwargs) -> pd.DataFrame:
    return pd.DataFrame({name: walk_forward(returns, fn, **kwargs).metrics()
                         for name, fn in strategies.items()}).T