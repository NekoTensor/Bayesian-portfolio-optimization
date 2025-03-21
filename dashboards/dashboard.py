
from dash import Dash, dcc, html, Input, Output
import dash_bootstrap_components as dbc
import pandas as pd
import plotly.express as px

# Load processed data
prices = pd.read_csv("data/processed/historical_prices_cleaned.csv", index_col=0, parse_dates=True)
returns = prices.pct_change().dropna()
cum_returns = (1 + returns).cumprod()

# Prepare figures
fig_cum = px.line(cum_returns.reset_index(), x="index", y=cum_returns.columns,
                  title="Cumulative Returns of Portfolio",
                  labels={"index": "Date", "value": "Cumulative Return"})
fig_cum.update_layout(title_font_size=24, xaxis_title="Date", yaxis_title="Cumulative Return")

fig_scatter = px.scatter_matrix(returns.reset_index(), dimensions=returns.columns,
                                title="Scatter Matrix of Asset Returns")
fig_scatter.update_layout(title_font_size=24)

# Create Dash app
app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
app.layout = html.Div([
    html.H1("Interactive Portfolio Dashboard", style={"textAlign": "center"}),
    dbc.Row([
        dbc.Col(dcc.Graph(id="cum_returns_graph", figure=fig_cum), md=6),
        dbc.Col(dcc.Graph(id="scatter_matrix_graph", figure=fig_scatter), md=6)
    ], style={"marginTop": "20px"}),
    html.Br(),
    html.Div([
        html.Label("Select Stress Scenario:"),
        dcc.Dropdown(
            id="stress_scenario",
            options=[
                {"label": "Baseline", "value": "Baseline"},
                {"label": "Market Crash", "value": "Market Crash"},
                {"label": "High Volatility", "value": "High Volatility"},
                {"label": "Combined Stress", "value": "Combined Stress"}
            ],
            value="Baseline",
            style={"width": "300px"}
        )
    ], style={"marginBottom": "20px", "textAlign": "center"}),
    dbc.Row([
        dbc.Col(dcc.Graph(id="stress_returns_graph"), md=12)
    ])
])

# Precompute stress simulations (similar to previous script)
import numpy as np
def monte_carlo(returns, scale_returns=1.0, scale_cov=1.0, num_sim=500, horizon=52):
    np.random.seed(42)
    mean_ret = returns.mean() * scale_returns
    cov_matrix = returns.cov() * scale_cov
    sims = []
    for _ in range(num_sim):
        sim = np.random.multivariate_normal(mean_ret, cov_matrix, horizon)
        sim_cum = (sim + 1).cumprod(axis=0)
        sims.append(sim_cum)
    return np.array(sims)

scenarios = {
    "Baseline": {"scale_returns": 1.0, "scale_cov": 1.0},
    "Market Crash": {"scale_returns": 0.5, "scale_cov": 1.0},
    "High Volatility": {"scale_returns": 1.0, "scale_cov": 2.0},
    "Combined Stress": {"scale_returns": 0.5, "scale_cov": 2.0}
}

simulations_dict = {}
for key, vals in scenarios.items():
    simulations_dict[key] = monte_carlo(returns, scale_returns=vals["scale_returns"], scale_cov=vals["scale_cov"], num_sim=500, horizon=52)

@app.callback(
    Output("stress_returns_graph", "figure"),
    Input("stress_scenario", "value")
)
def update_stress_graph(selected_scenario):
    sims = simulations_dict[selected_scenario]
    import plotly.graph_objects as go
    fig = go.Figure()
    for sim in sims[:50]:
        fig.add_trace(go.Scatter(
            x=list(range(sim.shape[0])),
            y=sim[:, 0],
            mode="lines",
            line=dict(color="gray", width=1),
            opacity=0.3,
            showlegend=False
        ))
    fig.update_layout(
        title=f"{selected_scenario} Scenario: Cumulative Returns (First Asset)",
        xaxis_title="Weeks",
        yaxis_title="Cumulative Return",
        title_font_size=24
    )
    return fig

if __name__ == "__main__":
    app.run_server(debug=True, port=8050)
