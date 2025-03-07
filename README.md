
![Portfolio Banner](https://img.freepik.com/free-vector/gradient-stock-market-concept_23-2149166910.jpg?t=st=1741372115~exp=1741375715~hmac=2c3eb4662650aa3999f60031125b9368682759e731d1286973f393473acd0ae2&w=2000)

# Portfolio Optimization with Bayesian Methods

Portfolio Optimization with Bayesian Methods is a comprehensive framework for optimizing asset allocation using advanced probabilistic techniques. In this project, we compare a classical naive mean-variance optimization approach with an advanced Bayesian Optimization method that leverages Gaussian Processes and acquisition functions to intelligently search for optimal portfolio weights. Our analysis includes extensive backtesting, stress testing, and interactive dashboard visualizations, demonstrating significant improvements in both performance and robustness.

---

## Table of Contents
- [Overview](#overview)
- [Theory and Methodology](#theory-and-methodology)
  - [Theoretical Foundations](#theoretical-foundations)
  - [Implementation Approach](#implementation-approach)
- [Results and Visualizations](#results-and-visualizations)
- [Dashboard and Documentation](#dashboard-and-documentation)
- [Installation and Usage](#installation-and-usage)
- [Repository Structure](#repository-structure)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)

---

## Overview
In modern financial markets, risk management and asset allocation are of utmost importance. Traditional portfolio optimization methods, like mean-variance optimization, have been widely used but do not account for uncertainties inherent in market data. Our project introduces Bayesian Optimization—a method that uses probabilistic modeling to estimate the distribution of outcomes—to create a more robust portfolio.

By integrating Gaussian Process regression and acquisition functions (e.g., Expected Improvement), our approach intelligently balances exploration and exploitation in the hyperparameter space. Extensive backtesting on historical market data shows that the Bayesian method can improve the Sharpe ratio by roughly 35% and reduce the maximum drawdown by approximately 34% compared to the naive approach.

*Key images in this repository include:*
- A raw data visualization to show the input data.
- A scatter matrix plot for exploring asset correlations.
- Cumulative returns and stress testing plots.
- Dashboard screenshots for real-time interactive analysis.

---

## Theory and Methodology

### Theoretical Foundations

Bayesian Optimization is a global optimization method that is particularly suited for expensive-to-evaluate functions. It uses a **Gaussian Process (GP)** to model the unknown objective function \( f(x) \) as:

$$
f(x) \sim GP(m(x), k(x,x'))
$$

where \( m(x) \) is the mean function (often assumed to be zero) and \( k(x,x') \) is a covariance kernel (typically, the Radial Basis Function kernel):

$$
k(x,x') = \sigma^2 \exp\left(-\frac{\|x-x'\|^2}{2l^2}\right).
$$

This formulation implies that points closer together in the parameter space are more correlated. The GP provides not only a mean prediction for any new input \( x \) but also an uncertainty (variance), which is crucial for decision-making in noisy environments.

An acquisition function, such as Expected Improvement (EI), is then used to determine the next point to evaluate:

$$
EI(x) = \mathbb{E}\left[\max\left(0, f(x) - f(x^*)\right)\right],
$$

where \( f(x^*) \) is the best observed value so far. This function effectively balances exploring areas with high uncertainty and exploiting areas with a high expected return.

Bayesian Optimization iteratively refines the GP model with new observations, thereby reducing the number of evaluations needed to find the optimal solution. This efficiency is especially attractive in financial applications where each evaluation (such as backtesting a portfolio) can be computationally expensive.

### Implementation Approach

Our project employs both a naive mean-variance approach and an advanced Bayesian Optimization framework. In the naive method, we solve the following optimization problem:

$$
\max_{w} \quad \frac{w^\top \mu}{\sqrt{w^\top \Sigma w}},
$$

subject to

$$
\sum_{i} w_i = 1.
$$

This problem is solved using quadratic programming, which provides a baseline portfolio allocation. However, the naive approach does not account for the uncertainty in return estimates.

In contrast, our Bayesian method uses a Gaussian Process to model the negative Sharpe ratio. We then utilize an acquisition function (Expected Improvement) to guide our search for optimal weights. This probabilistic method allows us to efficiently explore the weight space and account for uncertainty, leading to a more robust portfolio under volatile market conditions.

Our interactive notebooks include detailed code cells with extensive comments that explain every step of the process—from data collection to portfolio optimization and stress testing. For example, in our [03_Bayesian_Optimization.ipynb](notebooks/03_Bayesian_Optimization.ipynb) notebook, we provide a narrative on how the GP is constructed and updated, while our [04_Stress_Testing.ipynb](notebooks/04_Stress_Testing.ipynb) notebook illustrates various stress scenarios through interactive plots.

By leveraging these advanced techniques, our approach not only improves performance metrics (e.g., Sharpe ratio) but also enhances portfolio robustness, making it highly suitable for modern financial risk management.


---

## Results and Visualizations

Our backtesting experiments revealed the following key performance metrics:

- **Naive Portfolio:**  
  - Annualized Return: 15.01%  
  - Sharpe Ratio: 0.969  
  - Maximum Drawdown: -25.56%

- **Bayesian Portfolio:**  
  - Annualized Return: 15.98%  
  - Sharpe Ratio: 1.312  
  - Maximum Drawdown: -16.89%

These results demonstrate a 35% improvement in the Sharpe ratio and a 34% reduction in maximum drawdown for the Bayesian approach.

### Visualizations
Below are some key visualizations from the project:

- **Cumulative Returns:**
  
  ![Cumulative Returns](https://github.com/NekoTensor/assets/blob/main/Screenshot%20from%202025-03-08%2000-37-51.png?raw=true)  
  *Comparison of cumulative returns for the naive and Bayesian portfolios.*

- **Scatter Matrix:**
    
  ![Scatter Matrix](https://github.com/NekoTensor/assets/blob/main/Screenshot%20from%202025-03-07%2015-01-35.png?raw=true)  
  *Scatter matrix showing pairwise relationships among asset returns.*

- **Stress Testing Plots:**
  
  - Baseline: ![Baseline Stress](https://github.com/NekoTensor/assets/blob/main/baseline.png?raw=true)  
  - Market Crash: ![Market Crash Stress](https://github.com/NekoTensor/assets/blob/main/marketscrash.png?raw=true)  
  - High Volatility: ![High Volatility Stress](https://github.com/NekoTensor/assets/blob/main/highvolatility.png?raw=true)  
  - Combined Stress: ![Combined Stress](https://github.com/NekoTensor/assets/blob/main/combined%20stress.png?raw=true)



---

## Dashboard and Documentation
Our interactive dashboard, implemented in [05_Dashboard.ipynb](notebooks/05_Dashboard.ipynb) and the `dashboard.py` script, allows users to dynamically explore portfolio performance under different stress scenarios. The dashboard integrates interactive Plotly graphs and real-time simulation data to provide a comprehensive view of portfolio robustness.

Detailed documentation is provided in the `docs/` folder, which includes the final compiled report in PDF format. This documentation covers theoretical foundations, methodology, implementation details, and extensive experimental results, making it an excellent resource for both academic and professional audiences.

---

## Installation and Usage
To run the project, follow these steps:

1. **Clone the Repository:**
   ```bash
   git clone https://github.com/yourusername/portfolio-optimization.git
   cd portfolio-optimization
   
2. **Install Required Packages:**
   ```bash
   pip install -r requirements.txt

## Run the Notebooks
Open the notebooks in **Jupyter** or **Google Colab**:

- `01_Data_Collection.ipynb`
- `02_Naive_Optimization.ipynb`
- `03_Bayesian_Optimization.ipynb`
- `04_Stress_Testing.ipynb`
- `05_Dashboard.ipynb`

## Run the Scripts
Execute the following scripts in order:

```bash
python scripts/data_collection.py
python scripts/bayesian_optimization.py
python scripts/stress_testing.py
python scripts/visualizations.py
python scripts/dashboard.py
```

## Repository Structure
```
portfolio-optimization/
├── README.md
├── LICENSE
├── requirements.txt
├── setup.py
├── data/
│   ├── raw/
│   └── processed/
├── notebooks/
│   ├── 01_Data_Collection.ipynb
│   ├── 02_Naive_Optimization.ipynb
│   ├── 03_Bayesian_Optimization.ipynb
│   ├── 04_Stress_Testing.ipynb
│   └── 05_Dashboard.ipynb
├── scripts/
│   ├── data_collection.py
│   ├── bayesian_optimization.py
│   ├── stress_testing.py
│   ├── visualizations.py
│   └── dashboard.py
├── reports/
│   └── final_report.pdf
```

## Contributing
Contributions are welcome! If you have suggestions or improvements, please **fork** the repository and submit a **pull request**.
For major changes, open an **issue** first to discuss your proposed modifications.

## License
This project is licensed under the **MIT License**. See the [LICENSE](LICENSE) file for details.

## Contact
For questions or further information, please contact **Amey Kamble** at [ameyk@kgpian.iitkgp.ac.in](mailto:ameyk@kgpian.iitkgp.ac.in).
