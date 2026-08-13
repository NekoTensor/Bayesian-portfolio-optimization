"""Estimator tests.

Beyond mechanical checks, two of these verify the *statistical* claims the
estimators are chosen for: that Ledoit-Wolf shrinkage really does reduce
covariance estimation error in the short-window regime, and that Bayes-Stein
really does pull dispersed sample means together. Without those, the estimators
are just alternative arithmetic.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from portfolio.estimators import (
    ESTIMATORS,
    bayes_stein,
    black_litterman,
    ledoit_wolf,
    niw_posterior_predictive,
    sample_moments,
)


def make_panel(n_periods=200, n_assets=6, seed=3, corr=0.35):
    rng = np.random.default_rng(seed)
    cov = 0.0004 * (corr * np.ones((n_assets, n_assets)) + (1 - corr) * np.eye(n_assets))
    data = rng.multivariate_normal(np.full(n_assets, 0.0012), cov, size=n_periods)
    index = pd.date_range("2016-01-01", periods=n_periods, freq="W-FRI")
    return pd.DataFrame(data, index=index, columns=[f"A{i}" for i in range(n_assets)])


@pytest.mark.parametrize("name", list(ESTIMATORS))
def test_beliefs_are_well_formed(name):
    panel = make_panel()
    mu, cov = ESTIMATORS[name](panel)

    n = panel.shape[1]
    assert mu.shape == (n,) and np.isfinite(mu).all()
    assert cov.shape == (n, n) and np.isfinite(cov).all()
    np.testing.assert_allclose(cov, cov.T, atol=1e-12)
    assert np.linalg.eigvalsh(cov).min() > 0, "covariance must be positive definite"


@pytest.mark.parametrize("name", list(ESTIMATORS))
def test_estimators_are_deterministic(name):
    """Determinism is a precondition for the look-ahead poison test to be exact."""
    panel = make_panel()
    first, second = ESTIMATORS[name](panel), ESTIMATORS[name](panel)
    np.testing.assert_array_equal(first.mu, second.mu)
    np.testing.assert_array_equal(first.cov, second.cov)


@pytest.mark.parametrize("name", list(ESTIMATORS))
def test_singular_windows_are_rejected(name):
    """T <= N means a singular sample covariance; fail loudly rather than silently."""
    panel = make_panel(n_periods=5, n_assets=6)
    with pytest.raises(ValueError, match="singular|T >"):
        ESTIMATORS[name](panel)


def test_ledoit_wolf_beats_the_sample_covariance_in_short_windows():
    """The estimator's entire justification, checked against a known truth.

    Short window, many assets: the sample covariance is noisy and shrinkage
    should win on Frobenius error in the large majority of draws.
    """
    n_assets, n_periods, trials = 20, 60, 40
    rng = np.random.default_rng(101)
    true_corr = 0.4 * np.ones((n_assets, n_assets)) + 0.6 * np.eye(n_assets)
    vols = rng.uniform(0.01, 0.05, size=n_assets)
    true_cov = np.outer(vols, vols) * true_corr

    wins = 0
    for trial in range(trials):
        draw = np.random.default_rng(1000 + trial).multivariate_normal(
            np.zeros(n_assets), true_cov, size=n_periods
        )
        sample_err = np.linalg.norm(sample_moments(draw).cov - true_cov, "fro")
        shrunk_err = np.linalg.norm(ledoit_wolf(draw).cov - true_cov, "fro")
        wins += shrunk_err < sample_err

    assert wins >= 0.8 * trials, f"shrinkage won only {wins}/{trials} draws"


def test_shrinkage_intensity_decays_with_sample_size():
    """delta should fall roughly as 1/T once the target is misspecified.

    The data here follow a three-factor structure, so the constant-correlation
    target is genuinely wrong and shrinking toward it has a real cost. That makes
    the test informative: with more data the estimator must trust the sample more.
    Drawing from a constant-correlation covariance instead would send delta to 1
    at every T -- correct behaviour, but it would validate nothing.
    """
    n_assets = 12
    rng = np.random.default_rng(5)
    loadings = rng.normal(size=(n_assets, 3))
    true_cov = loadings @ loadings.T * 1e-4 + np.diag(rng.uniform(1e-4, 6e-4, n_assets))

    def implied_delta(draw: np.ndarray) -> float:
        sample = sample_moments(draw).cov
        shrunk = ledoit_wolf(draw).cov
        std = np.sqrt(np.diag(sample))
        corr = sample / np.outer(std, std)
        r_bar = (corr.sum() - n_assets) / (n_assets * (n_assets - 1))
        target = r_bar * np.outer(std, std)
        np.fill_diagonal(target, np.diag(sample))
        gap = target - sample
        denominator = (gap**2).sum()
        return float(((shrunk - sample) * gap).sum() / denominator) if denominator > 0 else 0.0

    deltas = []
    for n_periods in (40, 200, 2000):
        draws = [
            np.random.default_rng(100 + seed).multivariate_normal(
                np.zeros(n_assets), true_cov, size=n_periods
            )
            for seed in range(8)
        ]
        deltas.append(float(np.mean([implied_delta(d) for d in draws])))

    assert all(0.0 <= d <= 1.0 for d in deltas), deltas
    assert deltas[0] > deltas[1] > deltas[2], f"delta did not decay with T: {deltas}"
    assert deltas[0] > 0.25, f"short windows should shrink substantially, got {deltas[0]:.3f}"
    assert deltas[2] < 0.05, f"long windows should barely shrink, got {deltas[2]:.3f}"


def test_bayes_stein_pulls_means_toward_a_common_value():
    """Dispersed sample means must come out strictly less dispersed."""
    rng = np.random.default_rng(5)
    n_assets, n_periods = 8, 120
    # Deliberately spread the true means so sample dispersion is large.
    true_mu = np.linspace(-0.002, 0.006, n_assets)
    cov = 0.0009 * np.eye(n_assets)
    panel = rng.multivariate_normal(true_mu, cov, size=n_periods)

    plain = sample_moments(panel).mu
    shrunk = bayes_stein(panel).mu

    assert shrunk.std() < plain.std()
    # Shrinkage is toward a single point, so ordering is preserved.
    assert (np.argsort(plain) == np.argsort(shrunk)).all()


def test_niw_predictive_is_wider_than_the_plug_in_estimate():
    """Parameter uncertainty must widen the predictive distribution, never narrow it.

    The upper bound is the real content of this test. An inverse-Wishart prior
    scaled as ``psi_0 = scale_0 * nu_0`` rather than ``scale_0 * (nu_0 - p - 1)``
    is centred on several times the intended covariance, and the predictive
    inherits that as conservatism nobody asked for. Both bounds together pin the
    calibration; only asserting ">" would pass either way.
    """
    panel = make_panel(n_periods=80)
    plug_in = np.diag(sample_moments(panel).cov)
    predictive = np.diag(niw_posterior_predictive(panel).cov)

    ratio = predictive / plug_in
    assert (ratio > 1.0).all(), "predictive must be wider than the plug-in estimate"
    assert (ratio < 1.5).all(), f"predictive inflated by {ratio.max():.2f}x -- prior miscalibrated"


def test_niw_prior_influence_fades_as_data_accumulate():
    """With more observations the posterior mean must approach the sample mean."""
    short = make_panel(n_periods=60, seed=9)
    long = make_panel(n_periods=1200, seed=9)

    short_gap = np.abs(niw_posterior_predictive(short).mu - sample_moments(short).mu).mean()
    long_gap = np.abs(niw_posterior_predictive(long).mu - sample_moments(long).mu).mean()
    assert long_gap < short_gap


def test_black_litterman_view_tilts_toward_recent_winners():
    """A momentum view should raise the winners' expected returns relative to losers."""
    rng = np.random.default_rng(17)
    n_periods, n_assets = 160, 6
    panel = rng.normal(0.0, 0.02, size=(n_periods, n_assets))
    panel[-26:, 0] += 0.02  # asset 0 is an unambiguous recent winner
    panel[-26:, 1] -= 0.02  # asset 1 an unambiguous loser

    prior_free = sample_moments(panel).mu
    posterior = black_litterman(panel).mu

    assert posterior[0] - posterior[1] > prior_free[0] - prior_free[1] - 1e-9
    assert posterior[0] > posterior[1]


def test_black_litterman_rejects_views_longer_than_the_window():
    panel = make_panel(n_periods=30, n_assets=5)
    with pytest.raises(ValueError, match="view_lookback"):
        black_litterman(panel, view_lookback=40)
