# Project Architecture

This project is organized into several key components:

- **data/**: Contains functions for downloading and processing historical stock data.
- **notebooks/**: Jupyter notebooks that document the entire workflow, including exploratory analysis, Bayesian optimization, stress testing, advanced visualizations, and interactive dashboards.
- **scripts/**: Python scripts that encapsulate the core functionality:
  - `data_collection.py`: Downloads and preprocesses stock data.
  - `bayesian_optimization.py`: Implements Bayesian optimization (using Gaussian Process Regression and TPE).
  - `stress_testing.py`: Runs Monte Carlo simulations to stress test portfolio performance.
  - `visualizations.py`: Generates advanced static and interactive visualizations.
  - `dashboard.py`: Runs an interactive dashboard using Dash.
- **models/**: (Optional) Contains trained models (e.g., Bayesian model or reinforcement learning agent).
- **reports/**: (Optional) Stores final reports and analysis outputs.

This modular structure facilitates reproducibility, testing, and future enhancements.
