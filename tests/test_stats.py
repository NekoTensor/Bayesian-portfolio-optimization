"""Inference tests.

A hypothesis test is only useful if its size is right: a test that rejects 30% of
the time under the null would make every strategy in the study look significant.
Two tests here check calibration by simulation rather than trusting the algebra.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from portfolio.stats import (
    deflated_sharpe_ratio,
    jobson_korkie_memmel,
    probabilistic_sharpe_ratio,
    sharpe_difference_test,
    sharpe_ratio,
    stationary_bootstrap_ci,
)


def normal_returns(n=500, mean=0.001, vol=0.02, seed=0):
    return np.random.default_rng(seed).normal(mean, vol, size=n)


def test_sharpe_ratio_annualises_correctly():
    r = normal_returns(n=2000)
    per_period = sharpe_ratio(r, frequency=1)
    annual = sharpe_ratio(r, frequency=52)
    assert annual == pytest.approx(per_period * np.sqrt(52))


def test_identical_series_show_no_difference():
    r = pd.Series(normal_returns(seed=4))
    result = jobson_korkie_memmel(r, r)
    assert result.estimate == pytest.approx(0.0, abs=1e-12)
    assert result.p_value == pytest.approx(1.0)


def test_jobson_korkie_size_is_near_nominal():
    """Under the null of equal Sharpe, reject ~5% of the time at alpha = 0.05."""
    rejections = 0
    trials = 300
    for trial in range(trials):
        rng = np.random.default_rng(5000 + trial)
        # Same distribution, correlated the way two real strategies would be.
        shared = rng.normal(0.001, 0.015, size=400)
        a = shared + rng.normal(0, 0.008, size=400)
        b = shared + rng.normal(0, 0.008, size=400)
        rejections += jobson_korkie_memmel(a, b).p_value < 0.05

    rate = rejections / trials
    assert 0.01 <= rate <= 0.11, f"empirical size {rate:.3f} is far from 0.05"


def test_bootstrap_test_detects_a_genuine_difference():
    """Power check: a large, persistent Sharpe gap must be flagged."""
    rng = np.random.default_rng(21)
    n = 600
    shared = rng.normal(0.0, 0.015, size=n)
    weak = shared + rng.normal(0.0000, 0.006, size=n)
    strong = shared + rng.normal(0.0035, 0.006, size=n)  # much higher mean, same vol

    result = sharpe_difference_test(strong, weak, n_boot=399, random_state=1)
    assert result.estimate > 0
    assert result.p_value < 0.05, result


def test_bootstrap_test_does_not_flag_noise():
    rng = np.random.default_rng(33)
    n = 500
    shared = rng.normal(0.001, 0.015, size=n)
    a = shared + rng.normal(0, 0.007, size=n)
    b = shared + rng.normal(0, 0.007, size=n)

    result = sharpe_difference_test(a, b, n_boot=399, random_state=2)
    assert result.p_value > 0.10, result


def test_bootstrap_test_handles_degenerate_input_without_crashing():
    """Identical series give a zero standard error; return NaN rather than divide by it."""
    r = normal_returns(seed=8)
    result = sharpe_difference_test(r, r, n_boot=99)
    assert np.isnan(result.p_value) or result.p_value > 0.99


def test_short_samples_are_rejected():
    with pytest.raises(ValueError, match="at least 20"):
        sharpe_difference_test(normal_returns(n=10), normal_returns(n=10))


def test_stationary_bootstrap_interval_brackets_the_estimate():
    r = normal_returns(n=600, seed=12)
    point, lower, upper = stationary_bootstrap_ci(
        r, lambda x: sharpe_ratio(x, frequency=52), n_boot=299, random_state=3
    )
    assert lower < point < upper
    assert upper - lower > 0.1, "a 600-observation Sharpe interval should be visibly wide"


def test_stationary_bootstrap_lays_down_geometric_blocks():
    """Verify the resampling structure itself, not just that it returns a number.

    The vectorised index construction is easy to get subtly wrong in a way that
    still produces plausible intervals -- e.g. carrying the largest block origin
    forward instead of the most recent one, which silently destroys the block
    structure. Steps that continue a block advance the source index by exactly
    +1, so their frequency should be 1 - 1/mean_block.
    """
    series = np.arange(60, dtype=float)
    captured: list[np.ndarray] = []

    def spy(resample):
        captured.append(resample.copy())
        return 0.0

    for mean_block, expected in ((10.0, 0.90), (1.0, 1.0 / 60)):
        captured.clear()
        stationary_bootstrap_ci(
            series, spy, n_boot=300, mean_block=mean_block, random_state=1
        )
        idx = np.asarray(captured, dtype=int)
        steps = (idx[:, 1:] - idx[:, :-1]) % 60
        assert abs((steps == 1).mean() - expected) < 0.04, (
            f"mean_block={mean_block}: contiguous fraction "
            f"{(steps == 1).mean():.3f}, expected ~{expected:.3f}"
        )


def test_probabilistic_sharpe_ratio_is_a_probability():
    r = normal_returns(n=400, mean=0.002, seed=6)
    value = probabilistic_sharpe_ratio(r, benchmark_sr=0.0)
    assert 0.0 <= value <= 1.0
    # A clearly positive-Sharpe series should beat a zero benchmark convincingly.
    assert value > 0.9


def test_probabilistic_sharpe_ratio_falls_as_the_benchmark_rises():
    r = normal_returns(n=400, mean=0.002, seed=6)
    easy = probabilistic_sharpe_ratio(r, benchmark_sr=0.0)
    hard = probabilistic_sharpe_ratio(r, benchmark_sr=0.15)
    assert hard < easy


def test_deflated_sharpe_falls_as_more_configurations_are_tried():
    """The core of the correction: more trials, higher bar."""
    r = normal_returns(n=500, mean=0.0018, seed=15)
    values = [
        deflated_sharpe_ratio(r, n_trials=n, trial_sr_variance=0.0009) for n in (1, 5, 20, 100)
    ]
    assert values == sorted(values, reverse=True), values
    assert values[-1] < values[0]


def test_deflated_sharpe_with_one_trial_equals_the_undeflated_probability():
    r = normal_returns(n=300, seed=19)
    assert deflated_sharpe_ratio(r, n_trials=1, trial_sr_variance=0.001) == pytest.approx(
        probabilistic_sharpe_ratio(r, 0.0)
    )


def test_unaligned_series_are_rejected():
    with pytest.raises(ValueError, match="unaligned"):
        jobson_korkie_memmel(normal_returns(n=100), normal_returns(n=90))


def test_pandas_series_are_aligned_on_dates():
    """Overlapping but offset date ranges must be inner-joined, not zipped positionally."""
    index = pd.date_range("2020-01-03", periods=200, freq="W-FRI")
    a = pd.Series(normal_returns(n=200, seed=1), index=index)
    b = pd.Series(normal_returns(n=180, seed=2), index=index[20:])

    result = jobson_korkie_memmel(a, b)
    assert np.isfinite(result.p_value)
