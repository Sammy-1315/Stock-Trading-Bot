# Stock Trading Bot

This project implements a time-series forecasting pipeline for stock prices using historical data from Yahoo Finance. It leverages classic technical indicators such as moving averages (MA), Relative Strength Index (RSI), and MACD as features to predict the next day’s closing price for the SPY ETF.

Key features include:
- Feature engineering of popular technical indicators
- Time-series cross-validation with TimeSeriesSplit
- Training and evaluation of linear regression models (with options for other regressors)
- Visualization of predicted vs actual prices
- A simple backtesting framework simulating a basic trading strategy based on model predictions

This project serves as a foundation for exploring machine learning in financial time series and demonstrates how even simple linear models can provide meaningful predictive power in noisy market environments.
