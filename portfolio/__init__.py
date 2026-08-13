"""Estimator-vs-optimiser decomposition for mean-variance portfolio choice.

The package is deliberately small and separates the three things that a
portfolio backtest confuses most often:

``portfolio.estimators``
    Maps a training window onto a *belief* about next period -- a mean vector
    and a covariance matrix. Sample moments, Ledoit-Wolf shrinkage, Bayes-Stein
    and a Normal-Inverse-Wishart posterior predictive all live here.

``portfolio.strategies``
    Maps a belief onto weights, subject to a feasible set. Convex (SLSQP) and
    Gaussian-process (``gp_minimize``) search are two allocators over the *same*
    feasible set, which is what makes comparing them meaningful.

``portfolio.backtest``
    Maps an (estimator, allocator) pair onto a realised out-of-sample return
    series, charging transaction costs on turnover.

``portfolio.stats``
    Decides whether a difference between two such series is real.
"""

from .objectives import FeasibleSet, neg_sharpe, simplex_from_logits, turnover
from .data import load_prices, load_returns, to_returns

__version__ = "1.0.0"

__all__ = [
    "FeasibleSet",
    "load_prices",
    "load_returns",
    "neg_sharpe",
    "simplex_from_logits",
    "to_returns",
    "turnover",
]
