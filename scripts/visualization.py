#!/usr/bin/env python3
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

sns.set(style="whitegrid", context="talk", palette="deep")

def plot_cumulative_returns(prices):
    """
    Plots cumulative returns using Plotly Express.
    """
    returns = prices.pct_change().dropna()
    cum_returns = (1 + returns).cumprod()
    cum_returns_reset = cum_returns.reset_index().rename(columns={"index": "Date"})
    fig = px.line(cum_returns_reset, x="Date", y=cum_returns.columns,
                  title="Cumulative Returns of Portfolio",
                  labels={"Date": "Date", "value": "Cumulative Return"})
    fig.update_layout(title_font_size=24, xaxis_title="Date", yaxis_title="Cumulative Return")
    fig.show()

def plot_efficient_frontier(returns, tickers, num_portfolios=10000):
    """
    Generates and plots the efficient frontier using Monte Carlo simulation.
    """
    all_weights, ret_arr, risk_arr, sharpe_arr = [], [], [], []
    mean_returns = returns.mean()
    cov_matrix = returns.cov()
    for _ in range(num_portfolios):
        weights = np.random.random(len(tickers))
        weights /= np.sum(weights)
        all_weights.append(weights)
        ret_arr.append(np.dot(weights, mean_returns))
        risk_arr.append(np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights))))
        sharpe_arr.append(ret_arr[-1] / risk_arr[-1])
    fig = go.Figure(data=go.Scatter(
        x=risk_arr,
        y=ret_arr,
        mode='markers',
        marker=dict(
            size=5,
            color=sharpe_arr,
            colorscale='Viridis',
            showscale=True,
            colorbar=dict(title="Sharpe Ratio")
        )
    ))
    fig.update_layout(title="Efficient Frontier", xaxis_title="Risk (Volatility)", yaxis_title="Return", title_font_size=24)
    fig.show()

if __name__ == "__main__":
    # For testing, load a processed file.
    prices = pd.read_csv("data/processed/historical_prices_cleaned.csv", index_col=0, parse_dates=True)
    plot_cumulative_returns(prices)
    tickers = prices.columns.tolist()
    returns = prices.pct_change().dropna()
    plot_efficient_frontier(returns, tickers)
