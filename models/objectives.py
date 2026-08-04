# Portfolio weight parameterization and objective functions.
from __future__ import annotations
import numpy as np


def softmax_weights(params) -> np.ndarray:
    """Map unconstrained parameters onto long-only weights summing to 1."""
    x = np.asarray(params, dtype=float)
    x = x - np.max(x)          # shift for stability; softmax is shift-invariant
    e = np.exp(x)
    return e / e.sum()


def stick_breaking_weights(params) -> np.ndarray:
    """Bijective map from [0,1]^(n-1) onto the simplex -- one fewer dimension
    and no redundant direction, but order-dependent (asset 1 gets a different
    implied prior than asset n). Prefer softmax unless dimension matters."""
    v = np.asarray(params, dtype=float)
    w = np.empty(len(v) + 1)
    remaining = 1.0
    for i, vi in enumerate(v):
        w[i] = remaining * vi
        remaining -= w[i]
    w[-1] = remaining
    return w


def portfolio_stats(weights, mean_returns, cov_matrix, frequency=52):
    w = np.asarray(weights, dtype=float)
    ret = float(np.dot(w, mean_returns)) * frequency
    vol = float(np.sqrt(np.dot(w, np.dot(cov_matrix, w)))) * np.sqrt(frequency)
    return ret, vol


def turnover(weights, prev_weights) -> float:
    if prev_weights is None:
        return 0.0
    return float(np.abs(np.asarray(weights) - np.asarray(prev_weights)).sum())


def neg_sharpe(weights, mean_returns, cov_matrix, frequency=52, risk_free=0.0,
               prev_weights=None, cost_bps=0.0, turnover_penalty=0.0) -> float:
    """Negative annualized Sharpe, net of trading costs.

    `cost_bps` is charged on one-way turnover against `prev_weights` and
    annualized to match the numerator. `turnover_penalty` is a separate L1
    regularizer -- not a real cost, just a preference for stable portfolios.
    Leave at 0 to measure pure net-of-cost performance.
    """
    w = np.asarray(weights, dtype=float)
    ret, vol = portfolio_stats(w, mean_returns, cov_matrix, frequency)

    if vol <= 0:
        return np.inf

    if prev_weights is not None:
        tno = float(np.abs(w - np.asarray(prev_weights, dtype=float)).sum())
        ret -= tno * (cost_bps / 1e4) * frequency
        ret -= turnover_penalty * tno

    return -(ret - risk_free) / vol
