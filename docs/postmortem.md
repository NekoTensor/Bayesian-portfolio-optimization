# Postmortem: the original result was invalid

*Written August 2026, correcting the March 2025 version of this project.*

The first version of this repository claimed that Bayesian optimisation improved
the Sharpe ratio by 35% and cut maximum drawdown by 34% relative to naive
mean-variance optimisation. That claim was wrong. This document records what the
error was, why it produced a positive result almost automatically, what was
actually found once it was fixed, and what I should have done differently.

It is kept in the repository rather than quietly overwritten because a corrected
result is only worth as much as the correction is checkable. The original code is
preserved under [`legacy/`](../legacy/).

---

## 1. The error

`legacy/notebooks/3_bayesian_optimization.ipynb`, cell 2:

```python
mean_returns = returns.mean().values
cov_matrix   = returns.cov().values
```

`returns` is the entire 2012–2023 history. Both optimisers maximised the Sharpe
ratio implied by these full-sample moments. Then, in cell 7, performance was
evaluated with:

```python
port_performance(weights, returns)
```

— the same `returns` object used to fit the weights. There is no train/test split
anywhere in notebooks 2, 3 or 5.

**The reported numbers are in-sample fit statistics, not performance results.**
They describe how well each optimiser fitted data it had already seen.

`legacy/notebooks/5_back_Testing.ipynb`, which produced the numbers that reached
the README and the report abstract, is worse. Cell 7:

```python
naive_weights    = np.array([0.20, 0.15, 0.10, 0.15, 0.20, 0.10, 0.10])
bayesian_weights = np.array([0.1587, 0.0, 0.0, 0.2126, 0.0, 0.3065, 0.3222])
```

Both are hardcoded, copy-pasted from an earlier run. The notebook labelled
"backtesting" never called an optimiser. It applied two fixed weight vectors to
the full in-sample return series.

## 2. Why the error guaranteed a positive result

This is the part that matters more than the bug itself, and the part I did not
understand in March 2025.

The comparison was not merely uncontrolled — it was **rigged in favour of the
method being promoted**, in two independent ways.

**Search thoroughness masquerading as skill.** `gp_minimize` with 50 evaluations
searches the in-sample Sharpe surface far more thoroughly than a single SLSQP
solve. On a *fixed, already-observed* objective, the more thorough search lands
closer to the in-sample optimum by construction. So the Bayesian arm was
guaranteed to score higher on the reported metric under a null hypothesis of no
real edge whatsoever. The result was not evidence of robustness; it was a
measurement of how hard each optimiser searched. The original report's Discussion
section interpreted the gap as evidence of "robustness", which is close to exactly
backwards: the higher in-sample score reflects *more* overfitting, not less.

**Mismatched feasible sets.** The naive arm ran with `min_alloc=-0.5` — shorting
permitted, capped at ±50%. The Bayesian arm searched `Real(0, 1)` per asset:
long-only, and effectively uncapped in concentration after normalisation. These
are different optimisation problems. Any difference in outcome is partly, and
possibly entirely, attributable to the constraint set rather than to the method.

Two arms, two different problems, one already-seen objective, and unequal search
budgets. There was no configuration of that experiment that could have produced a
negative result.

## 3. A third error: the name

"Bayesian portfolio optimization" described Bayesian **optimisation** — a
black-box search technique — not Bayesian **statistics** applied to portfolio
construction. There was no prior over expected returns, no posterior update, no
shrinkage estimator, no posterior-predictive distribution. `pymc3` appeared in
`requirements.txt` and `setup.py` and was never imported.

This is not a pedantic distinction. Estimation error in `μ` and `Σ` is the central
difficulty in mean-variance portfolio choice, and Bayesian statistics is the
standard tool for attacking it. Naming the project after the technique that does
*not* address that problem, while using sample moments that do nothing about it,
inverted the actual state of the work.

## 4. What was done about it

| Defect | Fix |
|---|---|
| In-sample evaluation | `portfolio/backtest.py` — rolling-origin walk-forward. Weights for period $t$ use only data before $t$. Enforced by a test that corrupts future returns and asserts past weights are **bitwise** unchanged. |
| Mismatched feasible sets | `portfolio/objectives.py::FeasibleSet` — one constraint object passed to every allocator; every returned weight vector is checked against it. |
| Unequal search budgets | Both optimisers expose `search_budget`, so effort can be equalised and varied. |
| No baseline | 1/N is the benchmark every strategy is tested against, per DeMiguel, Garlappi & Uppal (2009). |
| No significance testing | `portfolio/stats.py` — Memmel's test, the Ledoit–Wolf studentised bootstrap, stationary-bootstrap intervals, and the Deflated Sharpe Ratio counting *every* configuration tried. |
| Not actually Bayesian | `portfolio/estimators.py` — Ledoit–Wolf shrinkage, Bayes–Stein, a Normal-Inverse-Wishart posterior predictive, and Black–Litterman. |
| Stress test measured one asset, assumed normality | `portfolio/stress.py` — portfolio-level VaR/CVaR under Gaussian, fitted Student-$t$ and block-bootstrap models, plus worst realised historical windows. |
| Nothing was runnable | `requirements.txt` listed `skopt` (correct name: `scikit-optimize`) and `pymc3` (dead on Python 3.10+), so `pip install -r requirements.txt` failed. `setup.py` pointed at a non-existent `src/`. Data lived at `data/Cleaned Data/` while every script read `data/processed/`. All fixed; the walk-forward module that existed in the last commit could not even import, because it referenced a `models/objectives.py` that was never written. |

## 5. What the honest experiment found

<!-- RESULTS:START -->
<!-- Generated by experiments/make_report_tables.py -- do not edit by hand. -->

The original experiment compared a convex max-Sharpe solve on sample moments
against `gp_minimize` on the same sample moments. In this codebase those are
`sample+convex` and `sample+gp_bayesopt`. Re-running that exact comparison
out-of-sample, on the same asset panel, with a shared feasible set:

| | v1 claim (in-sample) | corrected (walk-forward) |
|---|---|---|
| Naive Sharpe | 0.969 | 0.997 |
| "Bayesian" Sharpe | 1.312 | 0.923 |
| **Sharpe change** | **+35%** | **-7.4%** |
| Naive max drawdown | -25.56% | -17.71% |
| "Bayesian" max drawdown | -16.89% | -16.80% |
| **Drawdown change** | **-34%** | **-5.1%** |

**The sign of the headline result reverses.** Bayesian optimisation does not
improve the Sharpe ratio by 35%; it *reduces* it by 7% once the
weights are chosen without seeing the evaluation window. The drawdown reduction
survives in direction only, at 5% rather than 34%, and is far
inside the noise.

Widening to the full 15-configuration grid: the best strategy
reached a Sharpe of 1.041 against 0.919 for
1/N, and **0 of
15** configurations beat equal-weighting at the 5% level. The
median 95% confidence interval is
1.19 Sharpe
units wide — several times the size of the effect originally claimed. The v1
number was not merely wrong in magnitude; it was reported at a precision the data
could never have supported.

Full table and discussion in the [README](../README.md#results).

<!-- RESULTS:END -->

## 6. What I should have done differently

**Write the evaluation harness before the method.** The walk-forward code is not
complicated — it is about eighty lines. Had it existed first, the flawed result
could never have been generated, because there would have been nowhere to put a
full-sample mean.

**Treat a large improvement as a symptom.** A 35% Sharpe improvement from swapping
one optimiser for another, on the same inputs and the same objective, is not a
plausible finding. Convex optimisation solves the max-Sharpe problem essentially
exactly; a global search cannot do meaningfully better on a problem that is
already solved. The size of the reported effect was itself evidence of a bug, and
I read it as evidence of success.

**Include the baseline that can embarrass you.** 1/N was omitted from the original
comparison. It is the first thing a reviewer asks for, precisely because it so
often wins.

**Notice when the limitations section is missing the limitation.** The original
report listed six caveats, every one a generic property of Gaussian processes —
kernel sensitivity, cubic scaling, stationarity, local optima. Not one mentioned
the absence of an out-of-sample split. The closest was a final item on
"dependence on historical data", warning that the results might not transfer if
markets changed — which still assumes the backtest had measured something
transferable in the first place. It hadn't. Writing a limitations section by
listing generic properties of the technique, rather than by attacking one's own
experimental design, is a way of appearing rigorous without being it.

**Close the loop.** The three commits before this rewrite added a walk-forward
harness with a comment reading *"Hard to beat out-of-sample — include it or the
comparison isn't honest."* I understood the problem by then. But the harness was
never wired to anything, no notebook was re-run, and no README number changed —
and, as it turns out, the harness could not even be imported. Knowing the right
method and not running it leaves the false claim standing just as firmly as not
knowing.
