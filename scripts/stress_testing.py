#!/usr/bin/env python3
import numpy as np
import pandas as pd

def monte_carlo(returns, scale_returns=1.0, scale_cov=1.0, num_sim=1000, horizon=52):
    """
    Runs Monte Carlo simulations for a given stress scenario.
    """
    np.random.seed(42)
    mean_ret = returns.mean() * scale_returns
    cov_matrix = returns.cov() * scale_cov
    num_assets = returns.shape[1]
    sims = []
    for _ in range(num_sim):
        sim = np.random.multivariate_normal(mean_ret, cov_matrix, horizon)
        sim_cum = (sim + 1).cumprod(axis=0)
        sims.append(sim_cum)
    return np.array(sims)

def compute_var(simulations, asset_index=0, confidence_level=5):
    """
    Computes the Value at Risk (VaR) for a specified asset.
    """
    final_returns = simulations[:, -1, asset_index]
    var = np.percentile(final_returns, confidence_level)
    return var

if __name__ == "__main__":
    prices = pd.read_csv("data/processed/historical_prices_cleaned.csv", index_col=0, parse_dates=True)
    returns = prices.pct_change().dropna()
    
    scenarios = {
        "Baseline": {"scale_returns": 1.0, "scale_cov": 1.0},
        "Market Crash": {"scale_returns": 0.5, "scale_cov": 1.0},
        "High Volatility": {"scale_returns": 1.0, "scale_cov": 2.0},
        "Combined Stress": {"scale_returns": 0.5, "scale_cov": 2.0}
    }
    
    for name, params in scenarios.items():
        sims = monte_carlo(returns, scale_returns=params["scale_returns"], scale_cov=params["scale_cov"], num_sim=500, horizon=52)
        var_95 = compute_var(sims, asset_index=0, confidence_level=5)
        print(f"{name} Scenario - 5% VaR for first asset: {var_95:.4f}")
