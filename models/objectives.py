# Portfolio weight parameterization and objective functions
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
