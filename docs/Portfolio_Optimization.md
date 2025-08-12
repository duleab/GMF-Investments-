# Portfolio Optimization for GMF Investments

## Overview
This notebook implements Modern Portfolio Theory (MPT) to optimize asset allocation based on the forecasting results from the Time Series Forecasting phase. We'll generate the efficient frontier, calculate optimal portfolio weights, and maximize risk-adjusted returns for the three assets: TSLA, BND, and SPY.

## 1. Environment Setup & Library Imports

     Environment setup complete!
     Libraries imported successfully
    

## 2. Load Forecasting ResultsWe'll load the forecasting results from the previous phase to use as inputs for our portfolio optimization.

     Successfully loaded forecasting results from forecasting_results.json
    
    Forecasting Results Summary:
     Assets: ['TSLA', 'BND', 'SPY']
     Training Period: 2015-07-01 to 2023-12-31
     Testing Period: 2023-12-31 to 2025-07-31
    
    Model Performance Metrics:
    
     TSLA:
      ARIMA:
       MAE: 63.0369
       RMSE: 77.6568
       MAPE: 24.2990
       Directional_Accuracy: 46.7005
    
     BND:
      ARIMA:
       MAE: 1.7398
       RMSE: 2.0080
       MAPE: 2.4424
       Directional_Accuracy: 31.4721
    
     SPY:
      ARIMA:
       MAE: 86.0577
       RMSE: 95.9926
       MAPE: 15.0604
       Directional_Accuracy: 3.2995
    

## 3. Extract Expected Returns and Risk MetricsWe'll extract the expected returns and risk metrics from the forecasting results to use in our portfolio optimization.

    Expected Annual Returns:
     TSLA: 0.0143 (1.43%)
     BND: 0.0019 (0.19%)
     SPY: -0.0002 (-0.02%)
    

     Warning: Not enough parts in the second to last line of CI string for TSLA
     Warning: Not enough parts in the second to last line of CI string for BND
     Warning: Not enough parts in the second to last line of CI string for SPY
    Expected Annual Volatility:
     TSLA: 0.0000 (0.00%)
     BND: 0.0000 (0.00%)
     SPY: 0.0000 (0.00%)
    

## 4. Calculate Correlation MatrixWe'll estimate the correlation between assets based on the forecasting results.

    Correlation Matrix (historical):
              TSLA       BND       SPY
    TSLA  1.000000  0.110890  0.591795
    BND   0.110890  1.000000  0.214598
    SPY   0.591795  0.214598  1.000000
    


    
![png](Portfolio_Optimization_files/Portfolio_Optimization_10_1.png)
    


## 5. Calculate Covariance MatrixWe'll calculate the covariance matrix from the volatility estimates and correlation matrix.

    Covariance Matrix (annualized from historical returns):
              TSLA       BND       SPY
    TSLA  0.403437  0.004739  0.070084
    BND   0.004739  0.004527  0.002692
    SPY   0.070084  0.002692  0.034763
    

## 6. Portfolio Optimization FunctionsWe'll define functions to calculate portfolio returns, volatility, and Sharpe ratio.

## 7. Generate Efficient FrontierWe'll generate the efficient frontier by calculating the minimum volatility portfolio for different target returns.

    Maximum Sharpe Ratio Portfolio:
     Weights: {'TSLA': np.float64(1.0), 'BND': np.float64(0.0), 'SPY': np.float64(1.3010426069826053e-16)}
     Return: 0.0143
     Volatility: 0.6352
     Sharpe Ratio: -0.0089
    
    Minimum Volatility Portfolio:
     Weights: {'TSLA': np.float64(1.6371452804531117e-17), 'BND': np.float64(0.9453697715395576), 'SPY': np.float64(0.054630228460442416)}
     Return: 0.0018
     Volatility: 0.0665
     Sharpe Ratio: -0.2732
    

## 8. Visualize Efficient FrontierWe'll visualize the efficient frontier and highlight the optimal portfolios.



## 10. Export Optimization ResultsWe'll export the optimization results for use in the next phase (Strategy Validation).

     Successfully exported optimization results to portfolio_optimization_results.json
    




    True





    Successfully loaded forecasting results from forecasting_results.json
    
    Forecasting Results Summary:
    Assets: ['TSLA', 'BND', 'SPY']
    Training Period: 2015-07-01 to 2023-12-31
    Testing Period: 2023-12-31 to 2025-07-31
    
    Model Performance Metrics:
    
     TSLA:
      ARIMA:
       MAE: 63.0369
       RMSE: 77.6568
       MAPE: 24.2990
       Directional_Accuracy: 46.7005
    
     BND:
      ARIMA:
       MAE: 1.7398
       RMSE: 2.0080
       MAPE: 2.4424
       Directional_Accuracy: 31.4721
    
     SPY:
      ARIMA:
       MAE: 86.0577
       RMSE: 95.9926
       MAPE: 15.0604
       Directional_Accuracy: 3.2995
    

## 11. Conclusion
We've successfully implemented Modern Portfolio Theory to optimize asset allocation based on the forecasting results. The key findings are:

1.  **Efficient Frontier**: We generated the efficient frontier to visualize the risk-return tradeoff for different portfolio allocations.
2.  **Maximum Sharpe Ratio Portfolio**: We identified the portfolio with the highest risk-adjusted return (Sharpe ratio).
3.  **Minimum Volatility Portfolio**: We identified the portfolio with the lowest risk (volatility).
4.  **Asset Allocation**: We determined the optimal weights for each asset in our portfolio.

The optimization results have been exported for use in the next phase (Strategy Validation), where we'll backtest the performance of our optimized portfolios against benchmark strategies.


