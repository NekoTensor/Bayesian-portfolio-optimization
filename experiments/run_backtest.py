"""The main experiment: does the estimator or the optimiser drive performance?

Runs every (estimator, allocator) pair walk-forward over the same windows, with
the same feasible set and the same transaction costs, then tests each result
against the 1/N benchmark. Everything written to ``results/`` is produced here --
no number in the README is typed in by hand.

    python experiments/run_backtest.py

Design notes
------------
The grid is a factorial on purpose. Reading down a column tells you how much the
optimiser matters holding the belief fixed; reading across a row tells you how
much the belief matters holding the optimiser fixed. A single "naive vs Bayesian"
comparison -- what this project used to report -- cannot separate the two, which
is how it ended up crediting the optimiser for a difference in search space.

Every configuration tried is counted in ``n_trials`` for the Deflated Sharpe
Ratio, including the ones that lose. Counting only the winner is the mechanism by
which backtest overfitting enters the literature.

Configurations are independent, so they run in parallel across processes. The
Gaussian-process arms dominate the runtime -- each ``gp_minimize`` call refits the
surrogate after every evaluation -- so the wall-clock cost is roughly that of the
single slowest arm rather than the sum of all sixteen.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from portfolio.backtest import BacktestResult, make_strategy, walk_forward
from portfolio.data import load_returns
from portfolio.estimators import ESTIMATORS
from portfolio.objectives import FeasibleSet
from portfolio.stats import (
    deflated_sharpe_ratio,
    jobson_korkie_memmel,
    sharpe_difference_test,
    sharpe_ratio,
    stationary_bootstrap_ci,
)
from portfolio.strategies import ALLOCATORS

BENCHMARK = "equal_weight"


@dataclass(frozen=True)
class Spec:
    """A single cell of the design. Picklable, so workers can rebuild it."""

    name: str
    estimator: str
    allocator: str
    search_budget: int | None = None


def build_specs(gp_budget: int, convex_budget: int) -> list[Spec]:
    specs = [Spec(BENCHMARK, "sample", "equal_weight")]
    for estimator in ESTIMATORS:
        for allocator in ("min_variance", "convex", "gp_bayesopt"):
            budget = (
                gp_budget if allocator == "gp_bayesopt"
                else convex_budget if allocator == "convex"
                else None
            )
            specs.append(Spec(f"{estimator}+{allocator}", estimator, allocator, budget))
    return specs


def run_one(spec: Spec, backtest_kwargs: dict, max_weight: float) -> tuple[str, BacktestResult]:
    """Execute one cell. Module-level and argument-driven so it survives pickling."""
    warnings.filterwarnings("ignore")  # skopt is noisy about surrogate convergence

    returns = load_returns()
    feasible = FeasibleSet(n_assets=returns.shape[1], max_weight=max_weight)
    kwargs = {} if spec.search_budget is None else {"search_budget": spec.search_budget}

    strategy = make_strategy(
        ESTIMATORS[spec.estimator], ALLOCATORS[spec.allocator], feasible, **kwargs
    )
    return spec.name, walk_forward(returns, strategy, **backtest_kwargs)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-window", type=int, default=156, help="weeks of history per fit")
    parser.add_argument("--rebalance-every", type=int, default=13, help="holding period in weeks")
    parser.add_argument("--cost-bps", type=float, default=10.0, help="cost per unit traded")
    parser.add_argument("--max-weight", type=float, default=0.40, help="per-asset cap")
    parser.add_argument("--gp-budget", type=int, default=60, help="gp_minimize evaluations")
    parser.add_argument("--convex-budget", type=int, default=15, help="SLSQP restarts")
    parser.add_argument("--n-boot", type=int, default=2999, help="bootstrap resamples")
    parser.add_argument("--jobs", type=int, default=0, help="worker processes (0 = auto)")
    parser.add_argument("--expanding", action="store_true", help="expanding rather than rolling")
    parser.add_argument("--outdir", type=Path, default=Path(__file__).parent.parent / "results")
    args = parser.parse_args()

    returns = load_returns()
    specs = build_specs(args.gp_budget, args.convex_budget)
    jobs = args.jobs or min(len(specs), 8)

    print(f"panel      : {returns.shape[0]} weeks x {returns.shape[1]} assets "
          f"({returns.index.min().date()} to {returns.index.max().date()})")
    print(f"feasible   : long-only, weights capped at {args.max_weight:.0%}")
    print(f"walk-fwd   : {args.train_window}w train, {args.rebalance_every}w hold, "
          f"{args.cost_bps:.0f}bps cost, {'expanding' if args.expanding else 'rolling'}")
    print(f"grid       : {len(specs)} configurations across {jobs} processes\n")

    backtest_kwargs = dict(
        train_window=args.train_window,
        rebalance_every=args.rebalance_every,
        cost_bps=args.cost_bps,
        expanding=args.expanding,
    )

    started = time.time()
    results: dict[str, BacktestResult] = {}
    with ProcessPoolExecutor(max_workers=jobs) as pool:
        futures = {
            pool.submit(run_one, spec, backtest_kwargs, args.max_weight): spec
            for spec in specs
        }
        for future in as_completed(futures):
            name, result = future.result()
            results[name] = result
            print(f"  [{len(results):>2}/{len(specs)}] {name:32s} "
                  f"sharpe={result.metrics()['sharpe']:+.3f}  "
                  f"({time.time() - started:6.1f}s)", flush=True)

    # Restore the declared order so output is deterministic across runs.
    results = {spec.name: results[spec.name] for spec in specs}
    metrics = pd.DataFrame({n: r.metrics() for n, r in results.items()}).T
    metrics = metrics.sort_values("sharpe", ascending=False)

    # ---- Significance against the 1/N benchmark ---------------------------
    print(f"\ntesting {len(specs) - 1} strategies against {BENCHMARK} ...", flush=True)
    benchmark_returns = results[BENCHMARK].net_returns

    # Deflated Sharpe needs the spread of per-period Sharpe across everything tried.
    per_period = {n: sharpe_ratio(r.net_returns, frequency=1) for n, r in results.items()}
    trial_variance = float(np.var(list(per_period.values()), ddof=1))
    n_trials = len(specs)

    inference = {}
    for name, result in results.items():
        series = result.net_returns
        point, lower, upper = stationary_bootstrap_ci(
            series, lambda x: sharpe_ratio(x, frequency=52),
            n_boot=args.n_boot, random_state=7,
        )
        record = {
            "sharpe": point,
            "sharpe_ci_low": lower,
            "sharpe_ci_high": upper,
            "deflated_sharpe": deflated_sharpe_ratio(
                series, n_trials=n_trials, trial_sr_variance=trial_variance, frequency=52
            ),
        }
        if name != BENCHMARK:
            robust = sharpe_difference_test(
                series, benchmark_returns, frequency=52,
                n_boot=args.n_boot, random_state=11,
            )
            parametric = jobson_korkie_memmel(series, benchmark_returns, frequency=52)
            record.update({
                "sharpe_vs_benchmark": robust.estimate,
                "p_ledoit_wolf": robust.p_value,
                "p_jobson_korkie": parametric.p_value,
            })
        inference[name] = record

    inference_frame = pd.DataFrame(inference).T.sort_values("sharpe", ascending=False)

    # ---- Persist -----------------------------------------------------------
    outdir = args.outdir
    (outdir / "figures").mkdir(parents=True, exist_ok=True)

    metrics.to_csv(outdir / "metrics.csv")
    inference_frame.to_csv(outdir / "significance.csv")
    pd.DataFrame({n: r.net_returns for n, r in results.items()}).to_csv(
        outdir / "oos_returns.csv"
    )
    pd.DataFrame({n: r.turnover for n, r in results.items()}).to_csv(outdir / "turnover.csv")

    best = metrics.index[0]
    results[best].weights.to_csv(outdir / "weights_best.csv")
    results[BENCHMARK].weights.to_csv(outdir / "weights_benchmark.csv")

    config = {
        "generated_utc": pd.Timestamp.now(tz="UTC").isoformat(),
        "panel": {
            "assets": list(returns.columns),
            "start": str(returns.index.min().date()),
            "end": str(returns.index.max().date()),
            "n_periods": int(len(returns)),
        },
        "walk_forward": {**backtest_kwargs, "max_weight": args.max_weight},
        "search_budget": {"gp": args.gp_budget, "convex_restarts": args.convex_budget},
        "n_trials_for_dsr": n_trials,
        "trial_sr_variance": trial_variance,
        "n_boot": args.n_boot,
        "oos_periods": int(len(benchmark_returns)),
        "oos_start": str(benchmark_returns.index.min().date()),
        "oos_end": str(benchmark_returns.index.max().date()),
        "runtime_seconds": round(time.time() - started, 1),
    }
    (outdir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from make_figures import make_all_figures

    make_all_figures(results, metrics, inference_frame, outdir / "figures", BENCHMARK)

    # ---- Report ------------------------------------------------------------
    display = ["sharpe", "ann_return", "ann_vol", "max_drawdown", "avg_turnover", "cost_drag_ann"]
    print("\n=== Out-of-sample metrics ===")
    print(metrics[display].round(4).to_string())

    print("\n=== Sharpe vs 1/N, with 95% bootstrap intervals ===")
    cols = ["sharpe", "sharpe_ci_low", "sharpe_ci_high", "sharpe_vs_benchmark",
            "p_ledoit_wolf", "deflated_sharpe"]
    print(inference_frame[[c for c in cols if c in inference_frame]].round(4).to_string())

    p_values = inference_frame.get("p_ledoit_wolf")
    beat = int((p_values < 0.05).sum()) if p_values is not None else 0
    print(f"\nstrategies beating 1/N at the 5% level: {beat} of {len(specs) - 1}")
    print(f"total runtime: {time.time() - started:.1f}s")
    print(f"results written to {outdir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
