import numpy as np
import pandas as pd
from scipy.optimize import minimize
from skopt import gp_minimize
from skopt.space import Real
import plotly.express as px

class PortfolioOptimizer:
    def __init__(self, prices: pd.DataFrame, frequency: int = 52):
        # Initialize with price data (index must be datetime) and annualization factor (default=52 for weekly data)
        self.prices = prices
        self.returns = self.compute_returns(prices)
        self.assets = self.returns.columns.tolist()
        self.mean_returns = self.returns.mean().values
        self.cov_matrix = self.returns.cov().values
        self.frequency = frequency
        self.naive_opt = None
        self.bayesian_weights = None
        self.bayesian_sharpe = None

    @staticmethod
    def compute_returns(prices: pd.DataFrame) -> pd.DataFrame:
        # Compute returns from price data
        returns = prices.pct_change(fill_method=None).dropna()
        return returns

    def make_port(self, max_allocation=0.5, min_allocation=0.0,
                  risk_premium_up=0.5, risk_increment=0.005) -> pd.DataFrame:
        # Construct the efficient frontier using mean-variance optimization.
        n = len(self.mean_returns)
        risk_premiums = np.arange(0, risk_premium_up + risk_increment, risk_increment)
        portfolio_list = []
        bounds = [(min_allocation, max_allocation)] * n
        constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1}]

        def objective(x, rp):
            return 0.5 * np.dot(x, np.dot(self.cov_matrix, x)) - np.dot(rp * self.mean_returns, x)

        for rp in risk_premiums:
            res = minimize(objective, np.repeat(1/n, n), args=(rp,),
                           method='SLSQP', bounds=bounds, constraints=constraints)
            if res.success:
                w = np.round(res.x, 4)
                risk = np.sqrt(np.dot(w, np.dot(self.cov_matrix, w)))
                ret = np.dot(w, self.mean_returns)
                sharpe = ret / risk if risk > 0 else np.nan
                portfolio_list.append(np.concatenate((w, [risk, ret, sharpe])))

        cols = self.assets + ["Std.Dev", "Exp.Return", "Sharpe"]
        port_df = pd.DataFrame(portfolio_list, columns=cols)
        return port_df

    def naive_portfolio(self, max_alloc=0.5, min_alloc=-0.5) -> pd.Series:
        # Construct a naive optimal portfolio by maximizing the Sharpe ratio.
        eff = self.make_port(max_allocation=max_alloc, min_allocation=min_alloc)
        self.naive_opt = eff.loc[eff["Sharpe"].idxmax()]
        return self.naive_opt

    def neg_sharpe(self, weights) -> float:
        # Objective function: negative Sharpe ratio for minimization
        weights = np.array(weights) / np.sum(weights)
        ret = np.dot(weights, self.mean_returns)
        risk = np.sqrt(np.dot(weights.T, np.dot(self.cov_matrix, weights)))
        return -ret / risk

    def bayesian_portfolio(self, n_calls=50, random_state=42):
        # Construct an optimal portfolio using Bayesian optimization.
        dimensions = [Real(0, 1) for _ in self.assets]
        res = gp_minimize(lambda w: self.neg_sharpe(w), dimensions, n_calls=n_calls,
                          random_state=random_state)
        self.bayesian_weights = np.array(res.x) / np.sum(res.x)
        self.bayesian_sharpe = -res.fun
        return self.bayesian_weights, self.bayesian_sharpe, res

    @staticmethod
    def port_performance(opt_weights, test_returns: pd.DataFrame) -> dict:
        # Compute cumulative portfolio returns
        port_ret = test_returns.dot(opt_weights)
        tret = (port_ret + 1).cumprod().values
        return {"tret": tret, "tlab": test_returns.index, "weights": opt_weights}

    def port_summary(self, weights) -> tuple:
        # Compute portfolio performance metrics: annual return, risk, and Sharpe ratio
        port_ret = self.returns.dot(weights)
        ann_return = self.frequency * np.mean(port_ret)
        ann_risk = np.sqrt(self.frequency) * np.std(port_ret)
        sharpe_ratio = ann_return / ann_risk if ann_risk > 0 else np.nan
        return ann_return, ann_risk, sharpe_ratio

    def plot_cumulative_returns(self, weights, title="Cumulative Returns"):
        # Create an interactive Plotly line chart for cumulative portfolio returns.
        perf = self.port_performance(weights, self.returns)
        cum_df = pd.DataFrame({
            "Date": self.returns.index,
            "Cumulative Return": perf["tret"]
        })
        fig = px.line(cum_df, x="Date", y="Cumulative Return", title=title,
                      labels={"Cumulative Return": title})
        fig.update_layout(title_font_size=24, xaxis_title="Date", yaxis_title="Cumulative Return")
        return fig
