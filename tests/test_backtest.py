"""Backtest integrity tests.

The first test in this file is the one that matters most. The original version of
this project computed its headline Sharpe ratio by fitting weights on the full
sample and then evaluating on that same full sample, which is the single failure
mode most likely to make a backtest look good and mean nothing. A test that
*proves* the current harness cannot do that is worth more than any amount of
prose claiming it does not.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from portfolio.backtest import make_strategy, walk_forward
from portfolio.estimators import ESTIMATORS, sample_moments
from portfolio.objectives import FeasibleSet
from portfolio.strategies import ALLOCATORS, max_sharpe_convex


@pytest.fixture
def panel() -> pd.DataFrame:
    """A synthetic weekly panel with a known, benign structure."""
    rng = np.random.default_rng(20240817)
    n_periods, n_assets = 400, 5
    cov = 0.0004 * (0.3 * np.ones((n_assets, n_assets)) + 0.7 * np.eye(n_assets))
    data = rng.multivariate_normal(np.full(n_assets, 0.0015), cov, size=n_periods)
    index = pd.date_range("2015-01-02", periods=n_periods, freq="W-FRI")
    return pd.DataFrame(data, index=index, columns=[f"A{i}" for i in range(n_assets)])


@pytest.fixture
def feasible(panel) -> FeasibleSet:
    return FeasibleSet(n_assets=panel.shape[1], max_weight=0.5)


def test_future_data_cannot_influence_past_weights(panel, feasible):
    """Poison the future; every weight decided before it must be unchanged.

    If any estimator or allocator peeked forward -- via a full-sample mean, a
    global standardisation, a lookahead-shifted index -- corrupting later returns
    would move earlier weights. Bitwise equality is the correct assertion here:
    the strategy is deterministic, so anything other than an exact match is a leak.
    """
    strategy = make_strategy(sample_moments, max_sharpe_convex, feasible)
    kwargs = dict(train_window=104, rebalance_every=13, cost_bps=10.0)

    poison_from = 300

    # Build both panels through an identical construction path. Mutating a
    # DataFrame in place can split its internal block, which changes the memory
    # layout of *earlier* rows and with it the summation order inside .mean().
    # That perturbs weights at the 1e-8 level for reasons having nothing to do
    # with look-ahead, and would force this assertion down to a tolerance loose
    # enough to hide a small real leak. Constructing both the same way keeps the
    # comparison exact.
    values = panel.to_numpy(dtype=float)
    clean_values = values.copy()
    poisoned_values = values.copy()
    poisoned_values[poison_from:] *= -5.0  # violent, obvious corruption

    def rebuild(array):
        return pd.DataFrame(array, index=panel.index, columns=panel.columns)

    clean = walk_forward(rebuild(clean_values), strategy, **kwargs)
    poisoned = walk_forward(rebuild(poisoned_values), strategy, **kwargs)

    cutoff = panel.index[poison_from]
    unaffected = clean.weights.index < cutoff
    assert unaffected.sum() > 3, "test needs several pre-poison rebalances to be meaningful"

    np.testing.assert_array_equal(
        clean.weights.loc[unaffected].to_numpy(),
        poisoned.weights.loc[unaffected].to_numpy(),
    )

    # Realised returns before the corruption must also be untouched.
    pre = clean.net_returns.index < cutoff
    np.testing.assert_allclose(
        clean.net_returns[pre].to_numpy(), poisoned.net_returns[pre].to_numpy(), rtol=0, atol=0
    )


@pytest.mark.parametrize("estimator_name", list(ESTIMATORS))
@pytest.mark.parametrize("allocator_name", list(ALLOCATORS))
def test_weights_stay_inside_the_feasible_set(panel, feasible, estimator_name, allocator_name):
    """Every allocator, on every belief, must return an admissible portfolio."""
    kwargs = {"search_budget": 12} if allocator_name == "gp_bayesopt" else {}
    strategy = make_strategy(
        ESTIMATORS[estimator_name], ALLOCATORS[allocator_name], feasible, **kwargs
    )
    result = walk_forward(panel, strategy, train_window=104, rebalance_every=26)

    for date, row in result.weights.iterrows():
        w = row.to_numpy(float)
        assert feasible.contains(w), f"{estimator_name}+{allocator_name} left the set at {date}: {w}"


def test_training_window_never_includes_the_test_period(panel, feasible):
    """Directly inspect what the strategy is handed at each call."""
    seen: list[tuple[pd.Timestamp, pd.Timestamp]] = []

    def spy(train_returns, prev_weights=None):
        seen.append((train_returns.index[0], train_returns.index[-1]))
        return feasible.equal_weight()

    result = walk_forward(panel, spy, train_window=104, rebalance_every=13)

    for (_, train_end), test_start in zip(seen, result.weights.index):
        assert train_end < test_start, f"training data ran to {train_end}, test began {test_start}"


def test_expanding_window_grows_and_stays_causal(panel, feasible):
    lengths: list[int] = []

    def spy(train_returns, prev_weights=None):
        lengths.append(len(train_returns))
        assert train_returns.index[0] == panel.index[0], "expanding window must start at the origin"
        return feasible.equal_weight()

    walk_forward(panel, spy, train_window=104, rebalance_every=13, expanding=True)
    assert lengths == sorted(lengths) and lengths[-1] > lengths[0]


def test_transaction_costs_reduce_returns_and_scale_with_rate(panel, feasible):
    strategy = make_strategy(sample_moments, max_sharpe_convex, feasible)

    free = walk_forward(panel, strategy, train_window=104, rebalance_every=13, cost_bps=0.0)
    charged = walk_forward(panel, strategy, train_window=104, rebalance_every=13, cost_bps=25.0)

    assert charged.metrics()["ann_return"] < free.metrics()["ann_return"]
    # Gross series is cost-independent, so the two runs must agree on it exactly.
    np.testing.assert_allclose(
        free.gross_returns.to_numpy(), charged.gross_returns.to_numpy(), rtol=1e-12
    )
    # Total charge must equal traded notional times the rate.
    expected = charged.turnover.sum() * 25.0 / 1e4
    assert charged.meta["total_cost"] == pytest.approx(expected, rel=1e-12)


@pytest.mark.parametrize("cost_bps", [0.0, 5.0, 10.0, 25.0, 100.0])
def test_recosting_equals_refitting(panel, feasible, cost_bps):
    """``with_costs`` must reproduce a full refit at that cost, to the bit.

    The sensitivity sweep relies on this to avoid refitting the Gaussian-process
    arms once per cost level. If it ever drifts from a true refit, the sweep is
    silently reporting the wrong thing.
    """
    strategy = make_strategy(sample_moments, max_sharpe_convex, feasible)

    refit = walk_forward(panel, strategy, train_window=100, rebalance_every=13,
                         cost_bps=cost_bps)
    recost = walk_forward(panel, strategy, train_window=100, rebalance_every=13,
                          cost_bps=0.0).with_costs(cost_bps)

    pd.testing.assert_series_equal(recost.net_returns, refit.net_returns)
    pd.testing.assert_series_equal(recost.turnover, refit.turnover)
    assert recost.metrics()["sharpe"] == pytest.approx(refit.metrics()["sharpe"])
    assert recost.meta["total_cost"] == pytest.approx(refit.meta["total_cost"])


def test_recosting_rejects_negative_costs(panel, feasible):
    strategy = make_strategy(sample_moments, max_sharpe_convex, feasible)
    result = walk_forward(panel, strategy, train_window=100, rebalance_every=13)
    with pytest.raises(ValueError, match="non-negative"):
        result.with_costs(-1.0)


def test_equal_weight_turnover_is_only_drift(panel, feasible):
    """Rebalancing back to 1/N should trade far less than a full rotation."""
    strategy = make_strategy(sample_moments, ALLOCATORS["equal_weight"], feasible)
    result = walk_forward(panel, strategy, train_window=104, rebalance_every=13)

    assert result.turnover.iloc[0] == pytest.approx(1.0)  # initial build from cash
    assert (result.turnover.iloc[1:] < 0.25).all(), "drift-only turnover should be small"


def test_rejects_windows_that_leave_no_out_of_sample_data(panel, feasible):
    strategy = make_strategy(sample_moments, max_sharpe_convex, feasible)
    with pytest.raises(ValueError, match="no out-of-sample data"):
        walk_forward(panel, strategy, train_window=len(panel) + 1)


def test_metrics_are_internally_consistent(panel, feasible):
    strategy = make_strategy(sample_moments, max_sharpe_convex, feasible)
    result = walk_forward(panel, strategy, train_window=104, rebalance_every=13)
    m = result.metrics()

    r = result.net_returns
    assert m["sharpe"] == pytest.approx(r.mean() / r.std(ddof=1) * np.sqrt(52))
    assert m["total_return"] == pytest.approx((1.0 + r).prod() - 1.0)
    assert m["max_drawdown"] <= 0.0
    assert m["n_periods"] == len(r)
