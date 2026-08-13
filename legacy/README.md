# Legacy code (v1, March 2025)

This directory holds the original version of the project. It is kept deliberately,
not by accident: [`docs/postmortem.md`](../docs/postmortem.md) documents a specific
methodological error in this code, and that document is only checkable if the code
it describes is still here to read.

**None of it is imported by the current analysis.** Everything in `results/` is
produced by `portfolio/` and `experiments/`. If you are evaluating this repository,
read those; this directory is the "before" picture.

## What is here

| File | What it was | Why it was replaced |
|---|---|---|
| `v1_portfolio_optimizer.py` | The `PortfolioOptimizer` class | Fitted `mu`/`Sigma` on the full sample and evaluated on that same sample. Its `naive_portfolio()` allowed shorting (`min_alloc=-0.5`) while `bayesian_portfolio()` was long-only, so the two were never comparable. |
| `bayesian_optimization.py` | Standalone `gp_minimize` script | Searches `Real(0, 1)` per asset and normalises afterwards -- not a uniform parameterisation of the simplex. Superseded by `portfolio.strategies.max_sharpe_gp`. |
| `stress_testing.py` | Monte Carlo stress scenarios | Computed VaR for `asset_index=0` rather than for the portfolio, and drew from a multivariate normal, which cannot generate the tail losses being stress-tested for. Superseded by `portfolio.stress`. |
| `datacollection.py` | yfinance downloader | Superseded by `experiments/fetch_data.py`, which writes both raw and cleaned panels and reports what it dropped. |
| `visualization.py` | Plotly/seaborn charts | Efficient frontier by random sampling of 10,000 weight vectors -- fine as illustration, but the frontier is solved directly in `portfolio.strategies`. |
| `dashboards/` | Dash app + notebook | Plots the cumulative return of the *first asset*, labelled as the portfolio. Calls `app.run_server`, removed in Dash 3. |
| `notebooks/` | The original 1-5 analysis notebooks | Each begins with `from google.colab import files; files.upload()`, so none of them run locally or headlessly. Notebook 5, which produced the original README's headline numbers, hardcodes weights copied from an earlier run rather than calling an optimiser. |
| `PortfolioOptimization_v1_SUPERSEDED.pdf` | The March 2025 report | **Its abstract states the invalid result.** Kept so the postmortem can be checked against the document it corrects, and renamed so nobody reaches it expecting current findings. The current paper is built from `reports/LaTeX/` in CI. |

## The specific defect

`notebooks/3_bayesian_optimization.ipynb` computes

```python
mean_returns = returns.mean().values
cov_matrix   = returns.cov().values
```

over the entire 2012-2023 history, optimises weights against those moments, and
then evaluates performance with `port_performance(weights, returns)` -- the same
`returns` object. There is no train/test split in notebooks 2, 3, or 5.

The reported "Sharpe 0.969 -> 1.312, a 35% improvement" is therefore an in-sample
fit statistic, not a performance result. Worse, the improvement was close to
guaranteed: `gp_minimize` with 50 evaluations searches the in-sample Sharpe
surface more thoroughly than a single SLSQP solve, so it lands nearer the
in-sample optimum by construction -- which is *more* overfitting, not more
robustness.

See the postmortem for the full accounting and for what the honest out-of-sample
numbers turned out to be.

## Running it anyway

The scripts read `data/processed/historical_prices_cleaned.csv`, which now exists
(in v1 the data lived at `data/Cleaned Data/`, so the README's own instructions
failed on the second command). They need `plotly`, `dash` and `seaborn`, which are
optional extras:

```bash
pip install -e ".[dashboard]"
```
