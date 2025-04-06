import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import xgboost as xgb
from sklearn.model_selection import TimeSeriesSplit
from sklearn.linear_model import LinearRegression,Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

from sklearn.metrics import mean_squared_error,  mean_absolute_error


def set_up(stock_name):
    tick = yf.Ticker(stock_name)

    return tick.history(period = "5y")

def get_features():
    # tomorrows stock price
    data = set_up("SPY")
    data["Target"] = data["Close"].shift(-1)
    # Add features
    data['MA_10'] = data['Close'].rolling(window=10).mean()
    data['MA_50'] = data['Close'].rolling(window=50).mean()
    data['PCT_Change'] = data['Close'].pct_change()
    data['EMA_12'] = data['Close'].ewm(span=12, adjust=False).mean()
    data['EMA_26'] = data['Close'].ewm(span=26, adjust=False).mean()
    data['MACD'] = data['EMA_12'] - data['EMA_26']
    data['Signal_Line'] = data['MACD'].ewm(span=9, adjust=False).mean()
    
    # RSI Calculation
    delta = data['Close'].diff(1)
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=14).mean()
    avg_loss = loss.rolling(window=14).mean()
    rs = avg_gain / avg_loss
    data['RSI'] = 100 - (100 / (1 + rs))

    data.dropna(inplace=True)
    return data

def train_model(X, y):
    tscv = TimeSeriesSplit(n_splits=5)

    results = {
        "predictions": [],
        "actuals": [],
        "metrics": []
    }

    fold = 1
    for train_index, test_index in tscv.split(X):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        # can tune these hyperparamters

        model = LinearRegression()
        # model = RandomForestRegressor()
        # model = GradientBoostingRegressor()

        model.fit(X_train, y_train)

        predictions = model.predict(X_test)
        mse = mean_squared_error(y_test, predictions)
        mae = mean_absolute_error(y_test, predictions)

        results["predictions"].append(predictions)
        results["actuals"].append(y_test)
        results["metrics"].append({"fold": fold, "mse": mse, "mae": mae})

        print(f"Fold {fold}: MSE = {mse} MAE = {mae}")
        fold+=1
    

    return results
    
def plot_preds_actuals(results):

    predictions = np.concatenate(results["predictions"])
    actuals = np.concatenate(results["actuals"])
    # Create a scatter plot of predicted vs actual values
    # Create the plot
    plt.figure(figsize=(10, 6))

    # Plot the predictions (blue line)
    plt.plot(predictions, color='blue', label='Predictions')

    # Plot the actual values (red line)
    plt.plot(actuals, color='red', label='Actuals')

    plt.xlabel('Index')
    plt.ylabel('Price')
    plt.title('Predictions vs Actuals')

    plt.legend()

    plt.show()

def backtest(predictions, actuals):
    predictions = np.concatenate(predictions)
    actuals = np.concatenate(actuals)

    # Simulate trading with the precomputed predictions
    capital = 10000  # starting capital
    position = 0     # initial position (cash only)
    daily_returns = []  # list to hold daily returns

    # BUY_THRESHOLD = 0.02  # 2% predicted increase to trigger buy
    # SELL_THRESHOLD = -0.02  # 2% predicted decrease to trigger sell
    # STOP_LOSS = 0.95  # Exit if value drops to 95% of purchase price (5% loss)
    # TAKE_PROFIT = 1.15  # Exit if value rises to 110% of purchase price (10% profit)
    # buy_price = 0;

    # elementary trading strategy
    for i in range(1, len(predictions)):
        # Trading logic: Buy if the price will go up, Sell if it will go down
        if predictions[i] > predictions[i-1]:  # Model predicts price increase
            if position == 0:  
                position = capital / actuals[i-1]  
                capital = 0  
        elif predictions[i] < predictions[i-1]:  # Model predicts price decrease
            if position > 0:  
                capital = position * actuals[i-1]  
                position = 0  
    
        # Record daily return (capital + position value)
        daily_returns.append(capital + position * actuals[i])

    total_return = (daily_returns[-1] - daily_returns[0]) / daily_returns[0]

    print(total_return);

    # strategy 2:
    # Trading logic
    # for i in range(1, len(predictions)):
    #     # Calculate price change percentage
    #     price_change = (predictions[i] - predictions[i-1]) / predictions[i-1]

    #     # Handle Buy Logic
    #     if price_change > BUY_THRESHOLD and position == 0:
    #         # Buy if price change exceeds threshold and no position is held
    #         position = capital / actuals[i-1]  # Buy at the last actual price
    #         buy_price = actuals[i-1]  # Record the purchase price
    #         capital = 0  # All capital is invested

    #     # Handle Stop-Loss and Take-Profit Logic
    #     if position > 0:
    #         current_value = position * actuals[i-1]  # Current value of holdings
    #         # Stop-loss: Exit trade if value drops below threshold
    #         if actuals[i-1] < buy_price * STOP_LOSS:
    #             capital = current_value  # Sell all holdings
    #             position = 0  # Reset position

    #         # Take-profit: Exit trade if value exceeds target
    #         elif actuals[i-1] > buy_price * TAKE_PROFIT:
    #             capital = current_value  # Sell all holdings
    #             position = 0  # Reset position

    #     # Handle Sell Logic (independent of stop-loss and take-profit)
    #     if price_change < SELL_THRESHOLD and position > 0:
    #     # Sell if predicted decrease exceeds threshold
    #         capital = position * actuals[i-1]  # Sell at the last actual price
    #         position = 0  # Reset position

    #     daily_returns.append(capital + position * actuals[i])

    #     # Calculate performance metrics
    # total_return = (daily_returns[-1] - daily_returns[0]) / daily_returns[0]
    # print(total_return);


    # Plot the performance
    plt.plot(daily_returns)
    plt.title('Backtest Performance')
    plt.xlabel('Days')
    plt.ylabel('Portfolio Value')
    plt.show()
    
    


def main():
    data = get_features()
    
    features = ['MA_10', 'MA_50', 'PCT_Change', 'RSI', 'MACD', 'Signal_Line']
    X = data[features]
    y = data["Target"]
    
    results = train_model(X, y)
    
    plot_preds_actuals(results)
    backtest(results["predictions"], results["actuals"])


main()