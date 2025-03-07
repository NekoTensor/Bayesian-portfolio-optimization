#!/usr/bin/env python3
import pandas as pd
import numpy as np
from skopt import gp_minimize
from skopt.space import Real

def sharpe_ratio(weights, means, cov):
    weights = np.array(weights) / np.sum(weights)
    ret = np.dot(weights, means)
    risk = np.sqrt(np.dot(weights.T, np.dot(cov, weights)))
    return -ret / risk  # Negative for minimization

def optimize_portfolio(means, cov_matrix, dimensions):
    """
    Performs Bayesian optimization to maximize Sharpe ratio.
    """
    res = gp_minimize(lambda w: sharpe_ratio(w, means, cov_matrix),
                      dimensions, n_calls=50, random_state=42)
    optimal_weights = np.array(res.x) / np.sum(res.x)
    return optimal_weights, -res.fun

if __name__ == "__main__":
    # Load processed returns from file for testing (adjust path if needed)
    prices = pd.read_csv("data/processed/historical_prices_cleaned.csv", index_col=0, parse_dates=True)
    returns = prices.pct_change().dropna()
    mean_returns = returns.mean().values
    cov_matrix = returns.cov().values
    tickers = returns.columns.tolist()
    dimensions = [Real(0, 1) for _ in tickers]
    
    weights, max_sharpe = optimize_portfolio(mean_returns, cov_matrix, dimensions)
    print("Optimal Bayesian Portfolio Weights:", weights)
    print("Max Sharpe Ratio:", max_sharpe)
