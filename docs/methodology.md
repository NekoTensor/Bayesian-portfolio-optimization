# Methodology

## The question

> In mean-variance portfolio choice, how much of out-of-sample performance is
> attributable to the **estimator** (how beliefs about `μ` and `Σ` are formed)
> versus the **optimiser** (how weights are searched for, given those beliefs)?

This is worth asking because the two are routinely conflated. A paper reporting
that "Bayesian optimisation beats mean-variance" has usually changed the estimator,
the optimiser, *and* the constraint set simultaneously, and attributed the whole
difference to whichever component the title names. Separating them requires a
factorial design and an identical feasible set, which is what this study imposes.

The design also makes a null result informative. If the optimiser column has no
effect, that is a finding about where effort should go in portfolio construction.

## Data

Weekly adjusted closes for seven US-listed assets, 2012-01-08 to 2023-12-24
(625 return observations after differencing). Adjusted closes incorporate splits
and dividends, so returns are total returns.

The panel is committed to the repository rather than downloaded at runtime.
Yahoo Finance silently revises adjusted-close history, so a study that
re-downloads is not reproducible even against itself. `experiments/fetch_data.py`
rebuilds the panel when new data is actually wanted.

**On the universe.** Seven assets chosen without a documented rationale is a real
weakness, and it is the sharpest limitation of this study. After the fact the
composition is at least sensible — three high-idiosyncratic-variance technology
names (MSFT, GOOGL, AMZN), three lower-beta industrials and staples (MMM, GE,
HSY), and a short-duration Treasury ETF (SHY) as the low-volatility anchor — but
"sensible after the fact" is not "chosen on a stated principle beforehand". The
consequence is discussed under [Threats to validity](#threats-to-validity).

## Feasible set

All allocators optimise over

$$\mathcal{W} = \left\{ w \in \mathbb{R}^N : \sum_i w_i = 1,\; 0 \le w_i \le 0.40 \right\}$$

Long-only with a 40% per-asset cap. Two reasons. First, it is the constraint a
real mandate imposes. Second, unconstrained mean-variance on estimated moments
produces extreme long/short positions whose in-sample optimality does not survive
contact with new data — the cap is a crude but effective regulariser.

The critical property is that this set is *the same for every allocator*.
`portfolio.objectives.FeasibleSet` is passed to each one and every returned
weight vector is checked against it, so the earlier version's mistake — comparing
a shorting-permitted convex solver against a long-only GP search — cannot recur.

## The belief layer

Each estimator maps a training window onto a predictive mean and covariance.

**Sample moments.** Maximum likelihood. The baseline everything else must beat.

**Ledoit–Wolf (2004).** Shrinks the covariance toward a constant-correlation
target, $\hat\Sigma = \delta F + (1-\delta)S$, with $\delta$ chosen analytically to
minimise expected squared Frobenius error. The constant-correlation target is
used rather than scaled identity because equities share a dominant market factor:
pretending assets are uncorrelated is a worse prior than pretending they are
equally correlated.

**Bayes–Stein (Jorion, 1986).** Shrinks the mean toward $\mu_{\text{GMV}}$, the
expected return of the global minimum-variance portfolio, with intensity
$\lambda/(T+\lambda)$ where $\lambda$ depends on how dispersed the sample means
are relative to their standard errors. The covariance is inflated to reflect that
`μ` is estimated rather than known — the parameter-uncertainty premium that
plug-in mean-variance ignores.

**Normal-Inverse-Wishart posterior predictive.** A conjugate prior on
$(\mu, \Sigma)$, conditioned on the training window; the allocator receives the
*predictive* distribution of next period's return rather than a point estimate.
The predictive is multivariate-$t$ with $\nu_n - N + 1$ degrees of freedom, so its
covariance is strictly wider than the plug-in estimate. The prior is centred on a
common mean across assets — "absent evidence, assets earn the same" — which is
the same no-free-lunch stance that makes 1/N hard to beat.

**Black–Litterman (1992).** Reverse-optimises a reference portfolio into implied
equilibrium returns $\Pi = \delta \Sigma w_{\text{ref}}$, then blends in a
cross-sectional momentum view. The reference portfolio is equal-weight because
this panel carries no market capitalisations; with true caps this would be the
CAPM equilibrium, and with 1/N it is instead "the returns that would justify
holding everything equally". That substitution is an approximation, and it is
flagged rather than buried.

Every estimator is a pure function of its training window. That is what makes the
look-ahead guarantee testable.

## The search layer

**Equal weight (1/N).** Ignores the belief entirely. DeMiguel, Garlappi & Uppal
(2009) found no optimised policy in their sample consistently beat it out of
sample once estimation error was accounted for. A portfolio study without this
baseline is not measuring what it claims to.

**Minimum variance.** Uses `Σ`, ignores `μ`. A diagnostic: expected returns are
far harder to estimate than covariances, so if this wins, the `μ` estimates carry
no signal.

**Convex (SLSQP).** Multi-start sequential least-squares maximisation of the
Sharpe ratio, starts drawn from a Dirichlet.

**Gaussian-process Bayesian optimisation.** `gp_minimize` fits a GP surrogate and
selects each evaluation by maximising Expected Improvement. It searches in
unconstrained logit space; a softmax followed by projection maps that onto the
same feasible set SLSQP searches directly. The earlier version searched
$[0,1]^N$ and normalised afterwards, which is not a uniform parameterisation of
the simplex and biases the search toward the interior.

**Equalising search effort.** Both optimisers expose `search_budget`. Comparing
60 GP evaluations against a single SLSQP solve would confound the method with the
effort spent on it — and in-sample, the more thorough search *always* wins, which
is exactly how the original version manufactured its result.

## Walk-forward protocol

Weights held over period $t$ are a function of returns strictly before $t$:

- Training window: 156 weeks (3 years), rolling.
- Rebalance: every 13 weeks (quarterly).
- Between rebalances the portfolio is **held, not reset** — weights drift with
  relative performance, and turnover at the next rebalance is measured against
  the drifted holdings. Re-imposing targets every period would overstate both
  turnover and rebalancing alpha.
- Transaction costs: 10 bps per unit of traded notional, charged once to the
  period in which the trade occurs. Both sides count, so a full rotation costs
  20 bps.

This yields 469 out-of-sample weekly observations (2015 to 2023).

`tests/test_backtest.py::test_future_data_cannot_influence_past_weights` corrupts
all returns after a cutoff and asserts that every weight decided before it is
*bitwise* unchanged. That is a proof, not a claim.

## Inference

Point estimates are not conclusions. With 469 weekly observations the standard
error of an annualised Sharpe ratio near 1.0 is about 0.33 analytically
([Lo, 2002](https://doi.org/10.2469/faj.v58.n4.2453)), and the realised
stationary-bootstrap intervals in `results/significance.csv` imply ≈0.31 — a 95%
interval roughly 1.2 Sharpe units wide, which is wider than the entire spread
between the best and worst strategy in the study.

One caveat on reading those intervals. Overlapping marginal confidence intervals
do **not** by themselves imply that two strategies are statistically
indistinguishable: these strategies hold overlapping positions and their return
series are highly correlated, so the standard error of the *difference* is much
smaller than the standard error of either level. That is precisely why the
headline test is a paired one on the difference series rather than an eyeball
comparison of the intervals. Here the two agree — the best configuration's
difference from 1/N carries *p* = 0.52 — but they need not in general, and the
paired test is the one that governs.

- **Jobson–Korkie / Memmel (2003)** — closed-form test of equal Sharpe for
  dependent series. Fast; assumes iid normality.
- **Ledoit–Wolf (2008)** — studentised circular block bootstrap. Robust to the
  fat tails and autocorrelation returns actually exhibit. This is the headline
  test.
- **Politis–Romano (1994) stationary bootstrap** — confidence intervals on the
  Sharpe ratio itself.
- **Deflated Sharpe Ratio (Bailey & López de Prado, 2014)** — corrects for the
  fact that reporting the best of many configurations inflates the winner's
  Sharpe even under a null of no skill. **Every configuration run is counted**,
  including losers; counting only the winner is how backtest overfitting enters
  the literature.

`experiments/run_sensitivity.py` re-runs the comparison across 54 combinations of
training window, holding period, cost level and window type, and reports the
*frequency* with which each strategy beats 1/N — not its best case.

## Tail risk

`portfolio.stress` estimates portfolio-level VaR and CVaR three ways: Gaussian,
a fitted multivariate Student-$t$ (degrees of freedom by maximum likelihood), and
a stationary block bootstrap of the realised panel. The spread between them is a
measure of model risk, and it is reported rather than a single number.

`worst_historical_windows` reports the portfolio's worst realised stretches, which
require no calibration because they are facts about the data — unlike "multiply
the covariance by two", which is a number, not a crisis.

## Threats to validity

Stated plainly, because the previous version's limitations section listed generic
Gaussian-process caveats while omitting the one flaw that actually invalidated it.

1. **Universe size and selection.** Seven assets is small, and they were not
   chosen by a stated prior rule. With $N=7$ and $T=156$ the ratio $N/T$ is mild,
   which is precisely the regime where shrinkage has *least* to offer — so this
   study is a weak test of the estimators it compares. A wider universe would
   make estimation error bind and is the single most valuable extension.
2. **Survivorship.** All seven were liquid, listed names for the whole period.
   Any conclusion is conditional on that.
3. **One market regime.** 2015–2023 out-of-sample is dominated by a US equity
   bull market with two sharp drawdowns. It is one path, not a distribution over
   paths, and 469 weekly observations is a small sample for a Sharpe comparison.
4. **Cost model.** Proportional costs with no market impact, no bid-ask spread
   modelling, no capacity constraint. Adequate at these turnover levels; it would
   not be for a higher-frequency strategy.
5. **Momentum view specification.** The Black–Litterman view uses a 26-week
   lookback and a 50% damping factor. Neither was tuned — deliberately, since
   tuning them on this panel would reintroduce the selection bias the DSR exists
   to penalise — but neither is justified from first principles either.
6. **Single asset class.** Six equities and one Treasury ETF. Conclusions about
   diversification do not extend to multi-asset portfolios.

## References

- Bailey, D. & López de Prado, M. (2014). The Deflated Sharpe Ratio. *Journal of Portfolio Management*, 40(5).
- Black, F. & Litterman, R. (1992). Global Portfolio Optimization. *Financial Analysts Journal*, 48(5).
- DeMiguel, V., Garlappi, L. & Uppal, R. (2009). Optimal Versus Naive Diversification. *Review of Financial Studies*, 22(5).
- Jorion, P. (1986). Bayes-Stein Estimation for Portfolio Analysis. *JFQA*, 21(3).
- Ledoit, O. & Wolf, M. (2004). Honey, I Shrunk the Sample Covariance Matrix. *Journal of Portfolio Management*, 30(4).
- Ledoit, O. & Wolf, M. (2008). Robust Performance Hypothesis Testing with the Sharpe Ratio. *Journal of Empirical Finance*, 15(5).
- Memmel, C. (2003). Performance Hypothesis Testing with the Sharpe Ratio. *Finance Letters*, 1.
- Politis, D. & Romano, J. (1994). The Stationary Bootstrap. *JASA*, 89(428).
