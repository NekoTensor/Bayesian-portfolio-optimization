"""Feasible-set and objective tests.

These guard the property the whole comparison rests on: that both optimisers are
searching the same set. A bug that let one of them drift outside would recreate
exactly the apples-to-oranges comparison this rewrite exists to remove.
"""

from __future__ import annotations

import numpy as np
import pytest

from portfolio.objectives import (
    FeasibleSet,
    neg_sharpe,
    project_to_feasible,
    simplex_from_logits,
    turnover,
)


def test_rejects_caps_that_cannot_reach_full_investment():
    # Five assets capped at 10% each can hold at most 50% of the portfolio.
    with pytest.raises(ValueError, match="cannot sum to 1"):
        FeasibleSet(n_assets=5, max_weight=0.1)


def test_rejects_floors_that_overshoot_the_budget():
    with pytest.raises(ValueError, match="forces a sum"):
        FeasibleSet(n_assets=5, min_weight=0.3)


@pytest.mark.parametrize("max_weight", [0.25, 0.4, 1.0])
def test_projection_lands_in_the_set(max_weight):
    fs = FeasibleSet(n_assets=6, max_weight=max_weight)
    rng = np.random.default_rng(7)
    for _ in range(200):
        v = rng.normal(scale=3.0, size=6)
        w = project_to_feasible(v, fs)
        assert fs.contains(w), f"{v} -> {w}"


def test_projection_leaves_feasible_points_alone():
    fs = FeasibleSet(n_assets=4, max_weight=0.5)
    w = np.array([0.4, 0.3, 0.2, 0.1])
    np.testing.assert_allclose(project_to_feasible(w, fs), w, atol=1e-9)


def test_projection_is_the_nearest_feasible_point():
    """Check the Euclidean-projection claim against a constrained numerical solve."""
    from scipy.optimize import minimize

    fs = FeasibleSet(n_assets=5, max_weight=0.35)
    rng = np.random.default_rng(11)

    for _ in range(15):
        v = rng.normal(size=5)
        ours = project_to_feasible(v, fs)
        reference = minimize(
            lambda w: float(((w - v) ** 2).sum()),
            fs.equal_weight(),
            method="SLSQP",
            bounds=fs.bounds,
            constraints=[{"type": "eq", "fun": lambda w: w.sum() - 1.0}],
            options={"ftol": 1e-14, "maxiter": 800},
        )
        assert reference.success
        np.testing.assert_allclose(ours, reference.x, atol=1e-6)


def test_softmax_map_respects_the_cap():
    """Softmax alone cannot honour a per-asset cap; the composed map must."""
    fs = FeasibleSet(n_assets=4, max_weight=0.3)
    extreme = np.array([12.0, -6.0, -6.0, -6.0])  # softmax sends ~100% to asset 0
    w = simplex_from_logits(extreme, fs)
    assert fs.contains(w)
    assert w[0] <= 0.3 + 1e-9


def test_softmax_map_is_shift_invariant():
    fs = FeasibleSet(n_assets=5, max_weight=1.0)
    z = np.array([0.3, -1.2, 0.8, 2.0, -0.4])
    np.testing.assert_allclose(
        simplex_from_logits(z, fs), simplex_from_logits(z + 17.0, fs), atol=1e-12
    )


def test_turnover_counts_both_sides_of_the_trade():
    old = np.array([1.0, 0.0])
    new = np.array([0.0, 1.0])
    assert turnover(new, old) == pytest.approx(2.0)  # sell 100%, buy 100%
    assert turnover(old, old) == pytest.approx(0.0)
    assert turnover(old, None) == pytest.approx(1.0)  # building from cash


def test_neg_sharpe_matches_the_definition():
    mu = np.array([0.002, 0.001])
    cov = np.array([[4e-4, 1e-4], [1e-4, 9e-4]])
    w = np.array([0.6, 0.4])

    expected = -(52 * (w @ mu)) / np.sqrt(52 * (w @ cov @ w))
    assert neg_sharpe(w, mu, cov, frequency=52) == pytest.approx(expected)


def test_neg_sharpe_penalises_trading_when_costs_are_on():
    mu = np.array([0.002, 0.001])
    cov = np.array([[4e-4, 1e-4], [1e-4, 9e-4]])
    w = np.array([0.6, 0.4])
    prev = np.array([0.1, 0.9])  # large distance -> large turnover

    free = neg_sharpe(w, mu, cov, 52, cost_bps=0.0, prev_weights=prev)
    charged = neg_sharpe(w, mu, cov, 52, cost_bps=50.0, prev_weights=prev)
    assert charged > free  # negative Sharpe rises as costs bite


def test_degenerate_covariance_does_not_raise():
    """A zero-variance portfolio must be scored as unattractive, not crash the search."""
    value = neg_sharpe(np.array([1.0, 0.0]), np.array([0.01, 0.0]), np.zeros((2, 2)))
    assert np.isfinite(value) and value > 0
