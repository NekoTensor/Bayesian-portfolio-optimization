"""Robustness of the headline finding to the backtest's design choices.

A single walk-forward configuration is itself a researcher degree of freedom. If
the conclusion only holds at a 156-week training window, quarterly rebalancing
and 10 bps of cost, it is a property of those three numbers rather than of the
strategies. This script re-runs the comparison across a grid of plausible
settings and reports how often each strategy beats 1/N.

    python experiments/run_sensitivity.py

The output is deliberately a *frequency*, not a best case. Reporting the setting
where a strategy looks best is the same selection bias the Deflated Sharpe Ratio
in the main experiment exists to penalise.
"""

from __future__ import annotations

import argparse
import itertools
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from portfolio.backtest import make_strategy, walk_forward
from portfolio.data import load_returns
from portfolio.estimators import ESTIMATORS
from portfolio.objectives import FeasibleSet
from portfolio.strategies import ALLOCATORS

BENCHMARK = "equal_weight"

# A representative slice rather than the full 15-cell grid: one allocator family
# per estimator for the convex arm, plus two GP arms to check that the
# optimiser-irrelevance result is not an artefact of the base configuration.
SELECTED = [
    ("sample", "convex"),
    ("ledoit_wolf", "convex"),
    ("bayes_stein", "convex"),
    ("niw_predictive", "convex"),
    ("black_litterman", "convex"),
    ("ledoit_wolf", "min_variance"),
    ("sample", "gp_bayesopt"),
    ("ledoit_wolf", "gp_bayesopt"),
]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gp-budget", type=int, default=40)
    parser.add_argument("--max-weight", type=float, default=0.40)
    parser.add_argument("--outdir", type=Path, default=Path(__file__).parent.parent / "results")
    args = parser.parse_args()

    returns = load_returns()
    feasible = FeasibleSet(n_assets=returns.shape[1], max_weight=args.max_weight)

    strategies = {BENCHMARK: make_strategy(ESTIMATORS["sample"], ALLOCATORS["equal_weight"], feasible)}
    for est, alloc in SELECTED:
        kwargs = {"search_budget": args.gp_budget} if alloc == "gp_bayesopt" else {}
        strategies[f"{est}+{alloc}"] = make_strategy(
            ESTIMATORS[est], ALLOCATORS[alloc], feasible, **kwargs
        )

    train_windows = [104, 156, 260]
    holding_periods = [4, 13, 26]
    costs = [0.0, 10.0, 25.0]
    window_types = [False, True]  # rolling, expanding

    # Cost never enters the fit -- strategies are trained on returns alone -- so a
    # walk-forward is run once per (window, holding period, window type) and the
    # cost grid is applied to its realised turnover. Refitting per cost level
    # would triple the runtime to reproduce identical weights.
    fits = list(itertools.product(train_windows, holding_periods, window_types))
    print(f"{len(fits)} fits x {len(strategies)} strategies, "
          f"re-costed at {len(costs)} levels "
          f"({len(fits) * len(costs)} configurations)\n")

    records = []
    started = time.time()
    for i, (train_window, hold, expanding) in enumerate(fits, 1):
        for name, strategy in strategies.items():
            base = walk_forward(
                returns, strategy,
                train_window=train_window, rebalance_every=hold,
                cost_bps=0.0, expanding=expanding,
            )
            for cost in costs:
                metrics = base.with_costs(cost).metrics()
                records.append({
                    "train_window": train_window,
                    "rebalance_every": hold,
                    "cost_bps": cost,
                    "expanding": expanding,
                    "strategy": name,
                    "sharpe": metrics["sharpe"],
                    "ann_return": metrics["ann_return"],
                    "max_drawdown": metrics["max_drawdown"],
                    "avg_turnover": metrics["avg_turnover"],
                })
        print(f"  [{i:>2}/{len(fits)}] train={train_window:>3} hold={hold:>2} "
              f"{'expanding' if expanding else 'rolling  '} "
              f"({time.time() - started:6.1f}s)")

    frame = pd.DataFrame(records)
    args.outdir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.outdir / "sensitivity.csv", index=False)

    # --- How often does each strategy beat 1/N? ----------------------------
    wide = frame.pivot_table(
        index=["train_window", "rebalance_every", "cost_bps", "expanding"],
        columns="strategy", values="sharpe",
    )
    beats = wide.drop(columns=[BENCHMARK]).gt(wide[BENCHMARK], axis=0)

    summary = pd.DataFrame({
        "win_rate_vs_1overN": beats.mean(),
        "median_sharpe": wide.drop(columns=[BENCHMARK]).median(),
        "worst_sharpe": wide.drop(columns=[BENCHMARK]).min(),
        "best_sharpe": wide.drop(columns=[BENCHMARK]).max(),
    }).sort_values("win_rate_vs_1overN", ascending=False)
    summary.loc[BENCHMARK] = [
        float("nan"), wide[BENCHMARK].median(), wide[BENCHMARK].min(), wide[BENCHMARK].max()
    ]
    summary.to_csv(args.outdir / "sensitivity_summary.csv")

    print("\n=== Fraction of configurations in which each strategy beats 1/N ===")
    print(summary.round(4).to_string())
    print(f"\nSharpe spread across configurations for 1/N alone: "
          f"{wide[BENCHMARK].min():.3f} to {wide[BENCHMARK].max():.3f}")
    print(f"results written to {args.outdir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
