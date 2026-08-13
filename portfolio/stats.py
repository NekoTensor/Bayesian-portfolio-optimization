"""Deciding whether a difference between two return series is real.

A Sharpe ratio estimated on a few hundred observations is a random variable with
a wide distribution. On this panel the out-of-sample window is around 470 weekly
observations, where the standard error of a Sharpe ratio near 1.0 is roughly
0.15 -- so two strategies differing by 0.2 in Sharpe are not distinguishable, and
reporting such a gap as an improvement is reporting noise.

Four tools, in increasing order of how much they can tell you:

:func:`jobson_korkie_memmel`
    Closed-form test of equal Sharpe ratios. Fast, but assumes iid normal returns.
:func:`sharpe_difference_test`
    Ledoit-Wolf (2008) studentised block bootstrap. Robust to the fat tails and
    autocorrelation that returns actually exhibit; this is the headline test.
:func:`stationary_bootstrap_ci`
    Politis-Romano confidence intervals for any statistic of a return series.
:func:`deflated_sharpe_ratio`
    Bailey & Lopez de Prado (2014). Corrects for the fact that reporting the best
    of many configurations inflates the winner's Sharpe even under a null of no
    skill -- the single most under-used correction in applied backtesting.

References
----------
Memmel (2003), "Performance Hypothesis Testing with the Sharpe Ratio", Finance Letters 1.
Ledoit & Wolf (2008), "Robust Performance Hypothesis Testing with the Sharpe Ratio", JEF 15(5).
Politis & Romano (1994), "The Stationary Bootstrap", JASA 89(428).
Bailey & Lopez de Prado (2014), "The Deflated Sharpe Ratio", J. Portfolio Management 40(5).
"""

from __future__ import annotations

from typing import Callable, NamedTuple

import numpy as np
import pandas as pd
from scipy import stats as sps

__all__ = [
    "TestResult",
    "deflated_sharpe_ratio",
    "jobson_korkie_memmel",
    "probabilistic_sharpe_ratio",
    "sharpe_difference_test",
    "sharpe_ratio",
    "stationary_bootstrap_ci",
]

_EULER_MASCHERONI = 0.5772156649015329


class TestResult(NamedTuple):
    statistic: float
    p_value: float
    estimate: float
    std_error: float
    method: str

    def __repr__(self) -> str:  # pragma: no cover - display only
        return (
            f"{self.method}: diff={self.estimate:+.4f} (se={self.std_error:.4f}), "
            f"stat={self.statistic:+.3f}, p={self.p_value:.4f}"
        )


def _as_array(x) -> np.ndarray:
    a = x.to_numpy(dtype=float) if isinstance(x, (pd.Series, pd.DataFrame)) else np.asarray(x, float)
    a = a.squeeze()
    if a.ndim != 1:
        raise ValueError(f"expected a 1-D return series, got shape {a.shape}")
    return a[np.isfinite(a)]


def sharpe_ratio(returns, frequency: int = 1, risk_free: float = 0.0) -> float:
    """Sharpe ratio, annualised by ``sqrt(frequency)``.

    ``frequency=1`` returns the per-period value the tests below operate on.
    """
    r = _as_array(returns) - risk_free / frequency
    sd = r.std(ddof=1)
    return float(np.sqrt(frequency) * r.mean() / sd) if sd > 0 else np.nan


def _align(returns_a, returns_b) -> tuple[np.ndarray, np.ndarray]:
    """Align two series on their common index; both tests need paired samples."""
    if isinstance(returns_a, pd.Series) and isinstance(returns_b, pd.Series):
        joined = pd.concat([returns_a, returns_b], axis=1, join="inner").dropna()
        if joined.empty:
            raise ValueError("the two return series share no overlapping dates")
        return joined.iloc[:, 0].to_numpy(float), joined.iloc[:, 1].to_numpy(float)

    a, b = _as_array(returns_a), _as_array(returns_b)
    if len(a) != len(b):
        raise ValueError(f"unaligned series of lengths {len(a)} and {len(b)}")
    return a, b


def jobson_korkie_memmel(returns_a, returns_b, frequency: int = 1) -> TestResult:
    """Test ``H0: SR_a == SR_b`` for two dependent series, under iid normality.

    Memmel's (2003) correction of the Jobson-Korkie statistic. The variance term
    rewards correlation between the two strategies: two portfolios that move
    together can be distinguished on much smaller Sharpe gaps than two
    independent ones, because the shared market noise cancels in the difference.

    The normality assumption is the weak point -- use
    :func:`sharpe_difference_test` for the headline claim.
    """
    a, b = _align(returns_a, returns_b)
    t = len(a)
    if t < 10:
        raise ValueError(f"need at least 10 paired observations, got {t}")

    sr_a = a.mean() / a.std(ddof=1)
    sr_b = b.mean() / b.std(ddof=1)
    rho = float(np.corrcoef(a, b)[0, 1])

    variance = (
        2.0 - 2.0 * rho + 0.5 * (sr_a**2 + sr_b**2 - 2.0 * sr_a * sr_b * rho**2)
    ) / t
    se = float(np.sqrt(max(variance, 1e-300)))
    diff = float(sr_a - sr_b)
    z = diff / se

    return TestResult(
        statistic=float(z),
        p_value=float(2.0 * sps.norm.sf(abs(z))),
        estimate=float(diff * np.sqrt(frequency)),
        std_error=float(se * np.sqrt(frequency)),
        method="Jobson-Korkie / Memmel",
    )


def _hac_covariance(y: np.ndarray, bandwidth: int | None = None) -> np.ndarray:
    """Newey-West long-run covariance of the rows of ``y`` (T x k).

    Autocorrelation in squared returns is strong even when returns themselves
    look independent, so the ``r^2`` components below genuinely need this.
    """
    t = y.shape[0]
    centred = y - y.mean(axis=0)
    if bandwidth is None:
        bandwidth = int(np.floor(4.0 * (t / 100.0) ** (2.0 / 9.0)))
    bandwidth = max(0, min(bandwidth, t - 1))

    omega = (centred.T @ centred) / t
    for lag in range(1, bandwidth + 1):
        gamma = (centred[lag:].T @ centred[:-lag]) / t
        weight = 1.0 - lag / (bandwidth + 1.0)  # Bartlett
        omega += weight * (gamma + gamma.T)
    return omega


def _sharpe_diff_and_se(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    """Sharpe difference and its HAC standard error via the delta method.

    Parameterise each series by ``(mu, gamma)`` with ``gamma = E[r^2]``, so
    ``SR = mu / sqrt(gamma - mu^2)``. The gradient of the difference with respect
    to ``(mu_a, mu_b, gamma_a, gamma_b)`` combined with a Newey-West estimate of
    the long-run covariance of ``(r_a, r_b, r_a^2, r_b^2)`` gives the standard
    error without assuming any distribution.
    """
    t = len(a)
    mu_a, mu_b = a.mean(), b.mean()
    gamma_a, gamma_b = (a**2).mean(), (b**2).mean()

    var_a, var_b = gamma_a - mu_a**2, gamma_b - mu_b**2
    if var_a <= 0 or var_b <= 0:
        return np.nan, np.nan

    diff = mu_a / np.sqrt(var_a) - mu_b / np.sqrt(var_b)

    grad = np.array(
        [
            gamma_a / var_a**1.5,
            -gamma_b / var_b**1.5,
            -mu_a / (2.0 * var_a**1.5),
            mu_b / (2.0 * var_b**1.5),
        ]
    )
    y = np.column_stack([a, b, a**2, b**2])
    omega = _hac_covariance(y)
    variance = float(grad @ omega @ grad) / t
    return float(diff), float(np.sqrt(variance)) if variance > 0 else np.nan


def _circular_blocks(t: int, block_size: int, rng: np.random.Generator) -> np.ndarray:
    """Indices for one circular block-bootstrap resample of length ``t``."""
    n_blocks = int(np.ceil(t / block_size))
    starts = rng.integers(0, t, size=n_blocks)
    offsets = np.arange(block_size)
    return ((starts[:, None] + offsets[None, :]).ravel() % t)[:t]


def sharpe_difference_test(
    returns_a,
    returns_b,
    frequency: int = 1,
    n_boot: int = 4999,
    block_size: int | None = None,
    random_state: int = 0,
) -> TestResult:
    """Ledoit-Wolf (2008) studentised block bootstrap for ``H0: SR_a == SR_b``.

    Bootstrapping the difference alone would inherit the same distributional
    problems it is meant to avoid; studentising -- dividing each resampled
    difference by *its own* standard error -- gives a pivotal statistic whose
    bootstrap distribution converges faster and handles the skew and excess
    kurtosis of real return data.

    Blocks preserve serial dependence. The default block length follows the usual
    ``T^(1/3)`` rate, floored at 4 so that at weekly frequency roughly a month of
    dependence is retained inside each block.
    """
    a, b = _align(returns_a, returns_b)
    t = len(a)
    if t < 20:
        raise ValueError(f"need at least 20 paired observations, got {t}")

    if block_size is None:
        block_size = max(4, int(round(t ** (1.0 / 3.0))))
    block_size = int(np.clip(block_size, 1, t))

    diff, se = _sharpe_diff_and_se(a, b)
    if not np.isfinite(se) or se <= 0:
        return TestResult(np.nan, np.nan, diff, se, "Ledoit-Wolf bootstrap")

    observed = diff / se
    rng = np.random.default_rng(random_state)

    exceedances = 0
    valid = 0
    for _ in range(n_boot):
        idx = _circular_blocks(t, block_size, rng)
        boot_diff, boot_se = _sharpe_diff_and_se(a[idx], b[idx])
        if not np.isfinite(boot_se) or boot_se <= 0:
            continue
        valid += 1
        # Centre on the observed difference: the bootstrap world's null.
        if abs((boot_diff - diff) / boot_se) >= abs(observed):
            exceedances += 1

    p_value = (exceedances + 1) / (valid + 1) if valid else np.nan

    return TestResult(
        statistic=float(observed),
        p_value=float(p_value),
        estimate=float(diff * np.sqrt(frequency)),
        std_error=float(se * np.sqrt(frequency)),
        method=f"Ledoit-Wolf studentised bootstrap (block={block_size}, B={valid})",
    )


def stationary_bootstrap_ci(
    returns,
    statistic: Callable[[np.ndarray], float],
    n_boot: int = 4999,
    mean_block: float = 10.0,
    alpha: float = 0.05,
    random_state: int = 0,
) -> tuple[float, float, float]:
    """Politis-Romano stationary bootstrap interval for ``statistic``.

    Block lengths are geometric with mean ``mean_block`` rather than fixed, which
    keeps the resampled series stationary -- fixed blocks do not, and the
    resulting intervals are mildly miscalibrated.

    Returns ``(point_estimate, lower, upper)`` as a percentile interval.
    """
    r = _as_array(returns)
    t = len(r)
    if t < 20:
        raise ValueError(f"need at least 20 observations, got {t}")

    rng = np.random.default_rng(random_state)
    p = 1.0 / mean_block
    point = float(statistic(r))

    # Build every resample's index matrix at once. A resample is a sequence of
    # geometric-length blocks, so instead of stepping index-by-index in Python,
    # draw the block starts and lengths up front and lay the blocks down with
    # cumulative arithmetic -- same construction, ~100x faster, which is what
    # makes a 3000-resample interval practical on a laptop.
    fresh = rng.random((n_boot, t)) < p
    fresh[:, 0] = True  # every resample opens a block
    starts = rng.integers(0, t, size=(n_boot, t))

    # Index of the most recent block opening, forward-filled. Taking a running
    # maximum over *ordinals* is what carries the last True forward (ordinals
    # increase, so a later opening always wins); a running maximum over the
    # start values themselves would pick the largest origin rather than the
    # latest one.
    ordinal = np.broadcast_to(np.arange(t), (n_boot, t))
    last_opening = np.maximum.accumulate(np.where(fresh, ordinal, 0), axis=1)

    offset = ordinal - last_opening
    origin = np.take_along_axis(starts, last_opening, axis=1)
    idx = (origin + offset) % t

    draws = np.array([statistic(r[row]) for row in idx])

    finite = draws[np.isfinite(draws)]
    lower, upper = np.percentile(finite, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return point, float(lower), float(upper)


def probabilistic_sharpe_ratio(returns, benchmark_sr: float = 0.0, frequency: int = 1) -> float:
    """P(true Sharpe > ``benchmark_sr``), correcting for skew and kurtosis.

    ``benchmark_sr`` is expressed at the same frequency as ``returns``.
    """
    r = _as_array(returns)
    t = len(r)
    sr = sharpe_ratio(r, frequency=1)
    if not np.isfinite(sr) or t < 3:
        return np.nan

    bench = benchmark_sr / np.sqrt(frequency)
    skew = float(sps.skew(r, bias=False))
    kurt = float(sps.kurtosis(r, fisher=False, bias=False))  # raw: normal == 3

    denom = 1.0 - skew * sr + 0.25 * (kurt - 1.0) * sr**2
    if denom <= 0:
        return np.nan
    return float(sps.norm.cdf((sr - bench) * np.sqrt(t - 1) / np.sqrt(denom)))


def deflated_sharpe_ratio(returns, n_trials: int, trial_sr_variance: float, frequency: int = 1) -> float:
    """Bailey & Lopez de Prado's Deflated Sharpe Ratio.

    Searching over ``n_trials`` configurations and reporting the best one inflates
    the winner's Sharpe even when no configuration has any skill. The expected
    maximum of ``n_trials`` draws under that null becomes the benchmark the
    winner must clear, and the DSR is the probability it genuinely does.

    Parameters
    ----------
    n_trials
        How many configurations were actually tried -- honestly counted, including
        the ones that were discarded.
    trial_sr_variance
        Variance of the per-period Sharpe ratios *across* those trials.

    A DSR below 0.95 means the reported result is not distinguishable from the
    best of ``n_trials`` coin flips.
    """
    if n_trials < 1:
        raise ValueError("n_trials must be at least 1")
    if trial_sr_variance < 0:
        raise ValueError("trial_sr_variance must be non-negative")
    if n_trials == 1 or trial_sr_variance == 0:
        return probabilistic_sharpe_ratio(returns, 0.0, frequency)

    # Expected maximum of n_trials standard normals (Bailey-Lopez de Prado eq. 5).
    z1 = sps.norm.ppf(1.0 - 1.0 / n_trials)
    z2 = sps.norm.ppf(1.0 - 1.0 / (n_trials * np.e))
    expected_max = np.sqrt(trial_sr_variance) * (
        (1.0 - _EULER_MASCHERONI) * z1 + _EULER_MASCHERONI * z2
    )
    return probabilistic_sharpe_ratio(returns, expected_max * np.sqrt(frequency), frequency)
