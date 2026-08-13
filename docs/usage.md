# Usage

## Install

```bash
git clone https://github.com/NekoTensor/Bayesian-portfolio-optimization.git
cd Bayesian-portfolio-optimization
python -m pip install -e ".[dev]"
```

Python 3.10 or newer. The price panel is committed, so nothing needs downloading.

## Reproduce the published results

```bash
python experiments/run_backtest.py
```

Rewrites everything under `results/`. Takes roughly 35 minutes on a laptop; the
Gaussian-process arms dominate that time (five estimators × 37 rebalances × 60
surrogate evaluations each). Everything else runs in seconds.

To iterate quickly, cut the GP budget:

```bash
python experiments/run_backtest.py --gp-budget 15
```

## Check the robustness of the conclusion

```bash
python experiments/run_sensitivity.py
```

Re-runs the comparison across 54 combinations of training window, holding period,
cost level and rolling/expanding window, and reports how often each strategy beats
1/N rather than how good its best case looks.

Cost never enters the fit, so each walk-forward is run once per window
configuration and its realised turnover is re-charged at each cost level
(`BacktestResult.with_costs`). That is exact, not an approximation — a test
asserts it reproduces a full refit to the bit — and it keeps the sweep to
18 fits instead of 54.

## Run the tests

```bash
python -m pytest
```

The suite includes the look-ahead poison test, so a passing run is evidence that
the backtest is causal — not just that the code executes.

## Use the library directly

```python
from portfolio.backtest import make_strategy, walk_forward
from portfolio.data import load_returns
from portfolio.estimators import ledoit_wolf
from portfolio.objectives import FeasibleSet
from portfolio.strategies import max_sharpe_convex

returns = load_returns()
feasible = FeasibleSet(n_assets=returns.shape[1], max_weight=0.40)

strategy = make_strategy(ledoit_wolf, max_sharpe_convex, feasible)
result = walk_forward(returns, strategy, train_window=156, rebalance_every=13, cost_bps=10.0)

print(result.metrics())
```

Any callable with the signature `(train_returns) -> Belief` works as an estimator,
and any `(belief, feasible, prev_weights) -> weights` works as an allocator, so
adding a method means writing one function.

## Tail risk for a given allocation

```python
import numpy as np
from portfolio.stress import compare_var_models, worst_historical_windows

weights = result.weights.iloc[-1].to_numpy()
print(compare_var_models(returns, weights, horizon=13, level=0.05))
print(worst_historical_windows(returns, weights, window=13, top_k=5))
```

`compare_var_models` reports Gaussian, Student-$t$ and block-bootstrap estimates
side by side. The spread between them is the point.

## Refresh the data

```bash
python -m pip install "yfinance>=0.2.40"
python experiments/fetch_data.py --start 2012-01-01 --end 2023-12-31
```

Committed results will no longer match after this — Yahoo revises its adjusted
history. Re-run `run_backtest.py` to regenerate them.
