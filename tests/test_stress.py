"""Tail-risk tests.

The most important one here recovers a known degrees-of-freedom parameter from
simulated data. Without it, ``fit_student_t_dof`` is an optimiser returning a
number nobody has checked -- and a wrong ``nu`` would silently make the fat-tailed
model agree with the Gaussian one, quietly erasing the finding it exists to show.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from portfolio.stress import (
    bootstrap_var_cvar,
    compare_var_models,
    fit_student_t_dof,
    gaussian_var_cvar,
    portfolio_returns,
    student_t_var_cvar,
    tail_diagnostics,
    worst_historical_windows,
)


@pytest.fixture
def panel() -> pd.DataFrame:
    rng = np.random.default_rng(404)
    n_periods, n_assets = 500, 4
    cov = 0.0004 * (0.4 * np.ones((n_assets, n_assets)) + 0.6 * np.eye(n_assets))
    data = rng.multivariate_normal(np.full(n_assets, 0.001), cov, size=n_periods)
    index = pd.date_range("2014-01-03", periods=n_periods, freq="W-FRI")
    return pd.DataFrame(data, index=index, columns=[f"A{i}" for i in range(n_assets)])


@pytest.fixture
def weights(panel) -> np.ndarray:
    return np.repeat(1.0 / panel.shape[1], panel.shape[1])


def test_portfolio_returns_are_the_weighted_combination(panel, weights):
    series = portfolio_returns(panel, weights)
    np.testing.assert_allclose(series.to_numpy(), panel.to_numpy() @ weights)
    assert series.index.equals(panel.index)


def test_wrong_weight_length_is_rejected(panel):
    with pytest.raises(ValueError, match="expected 4 weights"):
        portfolio_returns(panel, np.array([0.5, 0.5]))


@pytest.mark.parametrize(
    "model", [gaussian_var_cvar, student_t_var_cvar, bootstrap_var_cvar]
)
def test_cvar_is_never_less_extreme_than_var(panel, weights, model):
    """CVaR averages the losses beyond VaR, so it must sit at or below it."""
    risk = model(panel, weights, horizon=13, level=0.05, n_sim=4000)
    assert risk.cvar <= risk.var
    assert risk.var < 0, "a 5% quantile of 13-week returns should be a loss"


def test_var_deepens_as_the_confidence_level_tightens(panel, weights):
    lenient = gaussian_var_cvar(panel, weights, level=0.10, n_sim=8000)
    strict = gaussian_var_cvar(panel, weights, level=0.01, n_sim=8000)
    assert strict.var < lenient.var


def test_student_t_dof_is_recovered_from_simulated_data():
    """Fit a known nu = 5; the estimate should land close."""
    true_dof, n_assets, n_periods = 5.0, 4, 4000
    rng = np.random.default_rng(77)
    cov = 0.0004 * (0.3 * np.ones((n_assets, n_assets)) + 0.7 * np.eye(n_assets))
    scale = cov * (true_dof - 2.0) / true_dof

    normal = rng.multivariate_normal(np.zeros(n_assets), scale, size=n_periods)
    chi2 = rng.chisquare(true_dof, size=n_periods)[:, None]
    draws = normal * np.sqrt(true_dof / chi2)

    estimated = fit_student_t_dof(pd.DataFrame(draws))
    assert 4.0 < estimated < 6.5, f"recovered nu={estimated:.2f}, expected around {true_dof}"


def test_high_dof_student_t_matches_the_gaussian(panel, weights):
    """As nu grows the t collapses onto the normal; the two VaRs should converge."""
    gaussian = gaussian_var_cvar(panel, weights, horizon=13, n_sim=30000, random_state=5)
    heavy_free = student_t_var_cvar(
        panel, weights, horizon=13, n_sim=30000, dof=200.0, random_state=5
    )
    assert abs(gaussian.var - heavy_free.var) < 0.01


def test_fat_tails_produce_deeper_losses_than_the_gaussian():
    """The headline claim of the module, on data with genuinely heavy tails."""
    n_assets, n_periods, dof = 3, 800, 3.5
    rng = np.random.default_rng(88)
    cov = 0.0004 * np.eye(n_assets)
    scale = cov * (dof - 2.0) / dof
    normal = rng.multivariate_normal(np.zeros(n_assets), scale, size=n_periods)
    chi2 = rng.chisquare(dof, size=n_periods)[:, None]
    panel = pd.DataFrame(normal * np.sqrt(dof / chi2))

    w = np.repeat(1.0 / n_assets, n_assets)
    gaussian = gaussian_var_cvar(panel, w, horizon=13, level=0.01, n_sim=40000, random_state=3)
    student = student_t_var_cvar(panel, w, horizon=13, level=0.01, n_sim=40000, random_state=3)

    assert student.var < gaussian.var, "fat tails must widen the 1% loss quantile"


def test_tail_diagnostics_flags_non_normality():
    rng = np.random.default_rng(91)
    normal = rng.normal(0, 0.02, size=2000)
    heavy = sps_t_sample(rng, dof=3.0, size=2000) * 0.01

    assert tail_diagnostics(normal)["normality_rejected_at_5pct"] is False
    heavy_stats = tail_diagnostics(heavy)
    assert heavy_stats["normality_rejected_at_5pct"] is True
    assert heavy_stats["excess_kurtosis"] > 1.0


def sps_t_sample(rng, dof, size):
    return rng.standard_normal(size) * np.sqrt(dof / rng.chisquare(dof, size))


def test_worst_windows_do_not_overlap(panel, weights):
    worst = worst_historical_windows(panel, weights, window=13, top_k=5)
    assert len(worst) == 5
    assert (worst["cumulative_return"] < 0).all()
    assert worst["cumulative_return"].is_monotonic_increasing

    starts = pd.to_datetime(worst["start"]).sort_values().to_numpy()
    gaps = np.diff(starts).astype("timedelta64[D]").astype(int)
    assert (gaps >= 13 * 7 - 7).all(), "windows should be effectively disjoint"


def test_compare_var_models_reports_all_three(panel, weights):
    frame = compare_var_models(panel, weights, horizon=13, n_sim=4000)
    assert len(frame) == 3
    assert {"var", "cvar", "var_vs_gaussian"} <= set(frame.columns)
    assert frame["var_vs_gaussian"].iloc[0] == pytest.approx(0.0)
