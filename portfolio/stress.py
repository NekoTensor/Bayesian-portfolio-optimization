"""Tail risk at the portfolio level.

The original stress test in this project did three things wrong, and each is
fixed here.

*It measured the wrong thing.* Value at Risk was computed for a single hardcoded
asset index rather than for the portfolio, so it never answered the question a
risk report exists to answer.

*It assumed normality.* Scenarios were multivariate-normal draws with the mean
scaled by 0.5 and the covariance by 2. Weekly equity returns have fat tails and
volatility clustering, and a Gaussian simulation is precisely the model that
cannot produce the losses you are stress-testing for. :func:`compare_var_models`
quantifies the size of that error rather than asserting it.

*Its scenarios were arbitrary.* "Multiply the covariance by two" is not a crisis;
it is a number. :func:`worst_historical_windows` reports what actually happened
to this portfolio in its worst realised stretches, which needs no calibration
because it is a fact about the data.

Three estimators of the same quantity are provided -- Gaussian, Student-t, and a
block bootstrap -- because the *gap between them* is the finding. If a Gaussian
model puts 5% VaR at -4% and the bootstrap puts it at -7%, the model risk exceeds
most of the differences between the allocation strategies being compared.
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np
import pandas as pd
from scipy import optimize, stats as sps
from scipy.special import gammaln

__all__ = [
    "TailRisk",
    "bootstrap_var_cvar",
    "compare_var_models",
    "fit_student_t_dof",
    "gaussian_var_cvar",
    "portfolio_returns",
    "student_t_var_cvar",
    "tail_diagnostics",
    "worst_historical_windows",
]


class TailRisk(NamedTuple):
    """Losses are reported as negative returns, so more negative is worse."""

    var: float
    cvar: float
    horizon: int
    level: float
    model: str


def portfolio_returns(returns: pd.DataFrame, weights) -> pd.Series:
    """Return series of a fixed-weight portfolio, rebalanced each period."""
    w = np.asarray(weights, dtype=float)
    if w.shape != (returns.shape[1],):
        raise ValueError(f"expected {returns.shape[1]} weights, got {w.shape}")
    return pd.Series(returns.to_numpy(dtype=float) @ w, index=returns.index, name="portfolio")


def tail_diagnostics(series) -> dict:
    """Evidence on whether a Gaussian assumption is defensible for this series."""
    r = np.asarray(series, dtype=float)
    r = r[np.isfinite(r)]
    jb_stat, jb_p = sps.jarque_bera(r)
    return {
        "n": int(len(r)),
        "mean": float(r.mean()),
        "std": float(r.std(ddof=1)),
        "skew": float(sps.skew(r, bias=False)),
        "excess_kurtosis": float(sps.kurtosis(r, fisher=True, bias=False)),
        "jarque_bera_stat": float(jb_stat),
        "jarque_bera_p": float(jb_p),
        "normality_rejected_at_5pct": bool(jb_p < 0.05),
    }


def worst_historical_windows(
    returns: pd.DataFrame, weights, window: int = 13, top_k: int = 5
) -> pd.DataFrame:
    """The ``top_k`` worst realised ``window``-period stretches for this portfolio.

    No distribution is assumed and no scenario is invented: these are drawdowns
    the portfolio would actually have suffered. Overlapping windows are filtered
    so the same crisis is not reported five times.
    """
    r = portfolio_returns(returns, weights)
    rolling = (1.0 + r).rolling(window).apply(np.prod, raw=True) - 1.0
    ranked = rolling.dropna().sort_values()

    selected: list[pd.Timestamp] = []
    positions = {date: i for i, date in enumerate(r.index)}
    for end_date in ranked.index:
        if all(abs(positions[end_date] - positions[d]) >= window for d in selected):
            selected.append(end_date)
        if len(selected) == top_k:
            break

    rows = []
    for end_date in selected:
        end = positions[end_date]
        rows.append({
            "start": r.index[end - window + 1].date(),
            "end": end_date.date(),
            "cumulative_return": float(ranked[end_date]),
            "worst_single_period": float(r.iloc[end - window + 1 : end + 1].min()),
        })
    return pd.DataFrame(rows).sort_values("cumulative_return").reset_index(drop=True)


def gaussian_var_cvar(returns: pd.DataFrame, weights, horizon: int = 13,
                      level: float = 0.05, n_sim: int = 20000,
                      random_state: int = 0) -> TailRisk:
    """VaR/CVaR under multivariate normality -- the model to be improved upon."""
    x = returns.to_numpy(dtype=float)
    rng = np.random.default_rng(random_state)
    draws = rng.multivariate_normal(x.mean(axis=0), np.cov(x, rowvar=False, ddof=1),
                                    size=(n_sim, horizon))
    w = np.asarray(weights, dtype=float)
    paths = (1.0 + draws @ w).prod(axis=1) - 1.0
    return _summarise(paths, horizon, level, "gaussian")


def fit_student_t_dof(returns: pd.DataFrame, bounds: tuple[float, float] = (2.5, 60.0)) -> float:
    """Maximum-likelihood degrees of freedom for a multivariate Student-t.

    The scale matrix is fixed at ``S * (nu - 2) / nu`` so the fitted distribution
    reproduces the sample covariance exactly and ``nu`` is left to describe the
    tails alone. Low ``nu`` means heavy tails; above roughly 30 the t is
    indistinguishable from a Gaussian.
    """
    x = returns.to_numpy(dtype=float)
    t, p = x.shape
    centred = x - x.mean(axis=0)
    cov = np.cov(x, rowvar=False, ddof=1)

    def negative_log_likelihood(nu: float) -> float:
        if nu <= 2.0:
            return np.inf
        scale = cov * (nu - 2.0) / nu
        sign, logdet = np.linalg.slogdet(scale)
        if sign <= 0:
            return np.inf
        quad = np.einsum("ij,jk,ik->i", centred, np.linalg.pinv(scale), centred)
        return -float(
            t * (gammaln((nu + p) / 2.0) - gammaln(nu / 2.0) - (p / 2.0) * np.log(nu * np.pi)
                 - 0.5 * logdet)
            - ((nu + p) / 2.0) * np.log1p(quad / nu).sum()
        )

    result = optimize.minimize_scalar(negative_log_likelihood, bounds=bounds, method="bounded")
    return float(result.x)


def student_t_var_cvar(returns: pd.DataFrame, weights, horizon: int = 13,
                       level: float = 0.05, n_sim: int = 20000,
                       dof: float | None = None, random_state: int = 0) -> TailRisk:
    """VaR/CVaR under a fitted multivariate Student-t.

    Same mean and covariance as the Gaussian model; only the tail thickness
    differs, which isolates the cost of the normality assumption.
    """
    x = returns.to_numpy(dtype=float)
    if dof is None:
        dof = fit_student_t_dof(returns)

    mean = x.mean(axis=0)
    scale = np.cov(x, rowvar=False, ddof=1) * (dof - 2.0) / dof

    rng = np.random.default_rng(random_state)
    normal = rng.multivariate_normal(np.zeros(len(mean)), scale, size=(n_sim, horizon))
    chi2 = rng.chisquare(dof, size=(n_sim, horizon))[..., None]
    draws = mean + normal * np.sqrt(dof / chi2)

    w = np.asarray(weights, dtype=float)
    paths = (1.0 + draws @ w).prod(axis=1) - 1.0
    risk = _summarise(paths, horizon, level, f"student_t(nu={dof:.1f})")
    return risk


def bootstrap_var_cvar(returns: pd.DataFrame, weights, horizon: int = 13,
                       level: float = 0.05, n_sim: int = 20000,
                       mean_block: float = 6.0, random_state: int = 0) -> TailRisk:
    """VaR/CVaR from a stationary block bootstrap of the realised return panel.

    Assumes no distribution at all. Resampling *blocks* of joint asset returns
    preserves both the cross-sectional correlation and the volatility clustering
    that make real drawdowns deeper than independent draws suggest -- which is
    exactly what the Gaussian and t models, both iid across time, cannot capture.
    """
    x = returns.to_numpy(dtype=float)
    t = len(x)
    w = np.asarray(weights, dtype=float)
    period_returns = x @ w

    rng = np.random.default_rng(random_state)
    p = 1.0 / mean_block

    idx = np.empty((n_sim, horizon), dtype=int)
    idx[:, 0] = rng.integers(0, t, size=n_sim)
    fresh = rng.random((n_sim, horizon)) < p
    jumps = rng.integers(0, t, size=(n_sim, horizon))
    for step in range(1, horizon):
        idx[:, step] = np.where(fresh[:, step], jumps[:, step], (idx[:, step - 1] + 1) % t)

    paths = (1.0 + period_returns[idx]).prod(axis=1) - 1.0
    return _summarise(paths, horizon, level, f"block_bootstrap(mean_block={mean_block:.0f})")


def _summarise(paths: np.ndarray, horizon: int, level: float, model: str) -> TailRisk:
    var = float(np.quantile(paths, level))
    tail = paths[paths <= var]
    cvar = float(tail.mean()) if tail.size else var
    return TailRisk(var=var, cvar=cvar, horizon=horizon, level=level, model=model)


def compare_var_models(returns: pd.DataFrame, weights, horizon: int = 13,
                       level: float = 0.05, n_sim: int = 20000,
                       random_state: int = 0) -> pd.DataFrame:
    """Run all three tail models side by side.

    The spread between rows is a measure of model risk. Reporting a single VaR
    number without it overstates how well the tail is understood.
    """
    models = [
        gaussian_var_cvar(returns, weights, horizon, level, n_sim, random_state),
        student_t_var_cvar(returns, weights, horizon, level, n_sim, random_state=random_state),
        bootstrap_var_cvar(returns, weights, horizon, level, n_sim, random_state=random_state),
    ]
    frame = pd.DataFrame([m._asdict() for m in models]).set_index("model")
    baseline = frame.loc[frame.index[0], "var"]
    frame["var_vs_gaussian"] = frame["var"] - baseline
    return frame
