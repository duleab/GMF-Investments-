# Time Series Forecasting for Portfolio Management Optimization

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange)](https://jupyter.org/)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)
[![Last Commit](https://img.shields.io/github/last-commit/duleab/GMF-Investments-?style=flat-square)](https://github.com/duleab/GMF-Investments-/commits/master)
[![Issues](https://img.shields.io/github/issues/duleab/GMF-Investments-?style=flat-square)](https://github.com/duleab/GMF-Investments-/issues)

> **A data-driven investment strategy leveraging advanced time series forecasting and portfolio optimization to maximize returns while managing risk.**

---

##  Project Overview

Guide Me in Finance (GMF) Investments is a financial advisory firm specializing in data-driven portfolio management.  
This project develops and implements advanced time series forecasting models to optimize investment portfolios, enhance client returns, and manage risk effectively.

---

##  Objectives

- **Data Analysis**: Comprehensive analysis of historical financial data using YFinance API  
- **Predictive Modeling**: Build robust forecasting models for market trend prediction  
- **Portfolio Optimization**: Optimize asset allocation based on forecasting insights  
- **Strategy Validation**: Validate investment strategies through rigorous backtesting  

---

##  Assets Under Analysis

Historical data (2015–2025) for three strategically selected assets:

| Asset | Symbol | Type | Purpose |
|-------|--------|------|----------|
| Tesla Inc. | TSLA | High-growth tech stock | Growth component |
| Vanguard Total Bond ETF | BND | Stable fixed income | Stability component |
| S&P 500 ETF | SPY | Market benchmark | Market exposure |

---

##  Technical Implementation

### Core Technologies
- **Python 3.8+**
- **Jupyter Notebooks**
- **YFinance**, **Pandas**, **NumPy**
- **Scikit-learn**, **TensorFlow/Keras**
- **Statsmodels**
- **Matplotlib**, **Seaborn**

### Key Features
- Advanced data cleaning & preprocessing  
- Stationarity testing & correlation analysis  
- ARIMA/SARIMA & LSTM model implementation  
- Portfolio optimization using Modern Portfolio Theory  
- Backtesting and performance evaluation  

---

##  Project Structure



##  Project Structure

```
Week11/
├── README.md
├── Notebook/
│ ├── Time_Series_Forecasting_for_Portfolio_Management_Optimization.ipynb
│ ├── Time Series Forecasting Models.ipynb
│ ├── Portfolio_Optimization.ipynb
│ └── Strategy_Validation.ipynb
├── data/
│ ├── gmf_financial_data_cleaned.xlsx
│ ├── forecasting_results.json
│ ├── portfolio_optimization_results.json
│ └── strategy_validation_results.json
├── docs/
│ ├── Portfolio_Optimization.md
│ ├── Strategy_Validation.md
│ ├── Time Series Forecasting Models.md
│ └── Time_Series_Forecasting_for_Portfolio_Management_Optimization.md
└── requirements.txt
```

##  Getting Started

### Prerequisites
```bash
Python 3.8 or higher
Jupyter Notebook or JupyterLab
```

### Installation

1. **Clone the repository*```bash
git clone https://github.com/yourusername/time-series-portfolio-optimization.git
cd time-series-portfolio-optimization
```

2. **Install dependencies*```bash
pip install -r requirements.txt
```

3. **Launch Jupyter Notebook*```bash
jupyter notebook
```

4. **Run the analysis*   - Start with `Time_Series_Forecasting_for_Portfolio_Management_Optimization.ipynb`
   - Follow the sequential execution of cells

##  Analysis Workflow

### Phase 1: Data Preprocessing & EDA
-  Extract and clean historical price data (2,535 trading days per asset)
-  Comprehensive exploratory data analysis with advanced visualizations
-  Missing value handling and outlier detection
-  Stationarity testing using Augmented Dickey-Fuller tests
-  Correlation analysis and risk metric calculations

### Phase 2: Time Series Forecasting
-  ARIMA/SARIMA statistical model implementation
-  LSTM deep learning model development
-  Model comparison using MAE, RMSE, MAPE metrics
-  Hyperparameter optimization and validation

### Phase 3: Portfolio Optimization
-  Modern Portfolio Theory implementation
-  Efficient frontier generation
-  Optimal weight calculation
-  Risk-adjusted return maximization

### Phase 4: Strategy Validation
-  Comprehensive backtesting framework
-  Performance comparison against benchmarks
-  Risk metric analysis and reporting

##  Key Results

### Data Quality Assessment
- **Dataset**: 7,605 total records across 3 assets
- **Coverage**: Jul 1, 2015 to Jul 31, 2025
- **Quality**: No missing values, comprehensive preprocessing

### Risk-Return Profiles
| Asset | Annual Return | Annual Volatility | Sharpe Ratio | Max Drawdown |
|-------|---------------|-------------------|--------------|---------------|
| TSLA  | Highest       | Highest (Growth)  | Variable     | Significant   |
| BND   | Stable        | Lowest (Defensive)| Consistent   | Minimal       |
| SPY   | Balanced      | Moderate          | Balanced     | Moderate      |

### Model Performance
- **ARIMA Model**: Strong statistical foundation, interpretable results
- **LSTM Model**: Captures complex patterns, superior for non-linear relationships
- **Ensemble Approach**: Combines strengths of both methodologies

##  Visual Highlights

Below are selected visuals exported from the notebooks. See more in `docs/`.

![Efficient Frontier](docs/images/Portfolio_Optimization_10_1.png)

![Forecast Diagnostics](docs/images/Time_Series_Forecasting_for_Portfolio_Management_Optimization_16_2.png)

![Forecast vs Actual](docs/images/Time_Series_Forecasting_for_Portfolio_Management_Optimization_18_0.png)

![Cumulative Performance](docs/images/Time Series Forecasting Models_21_1.png)

##  Configuration

### Model Parameters
- **ARIMA**: Auto-optimized using AIC/BIC criteria
- **LSTM**: 3-layer architecture with dropout regularization
- **Training Period**: 2015-2023 (2,091 samples)
- **Testing Period**: 2024-2025 (395 samples)

### Risk Management
- **VaR Calculation**: 95% and 99% confidence levels
- **Sharpe Ratio**: Risk-free rate assumption of 2%
- **Correlation Monitoring**: Dynamic correlation tracking




