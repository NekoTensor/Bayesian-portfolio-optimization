# Architecture

The package is organised around the distinction the study exists to measure. A
portfolio backtest chains four steps, and conflating any two of them is how
misleading results get produced:

```
price panel  ──▶  belief          ──▶  weights        ──▶  realised returns  ──▶  verdict
data.py           estimators.py        strategies.py       backtest.py            stats.py
                                       objectives.py                              stress.py
```

Each arrow is a pure function of what is to its left. That is what lets the
walk-forward harness guarantee causality and lets the experiment vary one stage
while holding the others fixed.

## `portfolio/`

| Module | Responsibility |
|---|---|
| `data.py` | Load the committed price panel; convert prices to returns. No network access. |
| `objectives.py` | `FeasibleSet` (the constraint set every allocator shares), the simplex projection and softmax parameterisation, turnover, and the Sharpe objective. |
| `estimators.py` | Training window → `Belief(mu, cov)`. Sample, Ledoit–Wolf, Bayes–Stein, NIW posterior predictive, Black–Litterman. |
| `strategies.py` | `Belief` + `FeasibleSet` → weights. 1/N, minimum variance, convex SLSQP, GP Bayesian optimisation. |
| `backtest.py` | Rolling-origin evaluation with weight drift and transaction costs. `make_strategy` composes an estimator with an allocator. |
| `stats.py` | Sharpe difference tests (Memmel; Ledoit–Wolf bootstrap), stationary-bootstrap intervals, Probabilistic and Deflated Sharpe. |
| `stress.py` | Portfolio-level VaR/CVaR under Gaussian, Student-$t$ and block-bootstrap models; worst realised windows. |

Two registries, `ESTIMATORS` and `ALLOCATORS`, let the experiment scripts declare
a grid rather than hard-code one.

## `experiments/`

Entry points. Everything in `results/` is produced by these and nothing else.

| Script | What it produces |
|---|---|
| `run_backtest.py` | The main factorial study → `metrics.csv`, `significance.csv`, `oos_returns.csv`, `config.json`, figures. |
| `run_sensitivity.py` | The same comparison across 54 backtest configurations → `sensitivity.csv`, `sensitivity_summary.csv`. |
| `make_figures.py` | Figure generation, imported by `run_backtest.py`. |
| `make_report_tables.py` | `results/` → the paper's tables and macros, and the README/postmortem results blocks. The documents hold no numbers of their own. |
| `fetch_data.py` | Optional: rebuild the price panel from Yahoo Finance. |

## `tests/`

Not incidental to the argument. `test_backtest.py` contains a look-ahead poison
test that corrupts future returns and asserts past weights are bitwise unchanged;
`test_estimators.py` verifies that Ledoit–Wolf shrinkage actually reduces
estimation error against a known covariance; `test_stats.py` checks the empirical
size of the hypothesis tests by simulation. Those three are the reason the
results can be believed.

`test_report.py` guards the documents rather than the code: it fails if the paper
cites a macro the generator no longer emits, if a generated number disagrees with
`results/`, or if regenerating would change a committed file. That is the drift
the previous version of this project died of, made mechanical.

## `legacy/`

The March 2025 version, preserved so that [`postmortem.md`](postmortem.md) can be
checked against the code it describes. Nothing in the current analysis imports it.
