"""Weight-selection strategies with a common signature:
    fit(train_returns, prev_weights) -> weights
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from .objectives import neg_sharpe, softmax_weights


def equal_weight(train_returns, prev_weights=None) -> np.ndarray:
    """1/N. Hard to beat out-of-sample -- include it or the comparison isn't honest."""
    n = train_returns.shape[1]
    return np.repeat(1.0 / n, n)


def mean_variance(train_returns, prev_weights=None, frequency=52,
                  cost_bps=0.0, max_weight=1.0) -> np.ndarray:
    mu = train_returns.mean().values
    cov = train_returns.cov().values
    n = len(mu)
    bounds = [(0.0, max_weight)] * n
    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]

    def obj(w):
        return neg_sharpe(w, mu, cov, frequency, prev_weights=prev_weights,
                          cost_bps=cost_bps)

    best, best_val = None, np.inf
    rng = np.random.default_rng(0)
    for x0 in [np.repeat(1.0 / n, n)] + [rng.dirichlet(np.ones(n)) for _ in range(4)]:
        res = minimize(obj, x0, method="SLSQP", bounds=bounds, constraints=constraints)
        if res.success and res.fun < best_val:
            best, best_val = res.x, res.fun

    if best is None:
        return equal_weight(train_returns)
    w = np.clip(best, 0, None)
    return w / w.sum()


def bayesian(train_returns, prev_weights=None, frequency=52, cost_bps=0.0,
             n_calls=50, random_state=42, bound=3.0) -> np.ndarray:
    from skopt import gp_minimize
    from skopt.space import Real

    mu = train_returns.mean().values
    cov = train_returns.cov().values

    def obj(params):
        w = softmax_weights(params)
        return neg_sharpe(w, mu, cov, frequency, prev_weights=prev_weights,
                          cost_bps=cost_bps)

    space = [Real(-bound, bound) for _ in range(len(mu))]
    res = gp_minimize(obj, space, n_calls=n_calls, random_state=random_state)
    return softmax_weights(res.x)