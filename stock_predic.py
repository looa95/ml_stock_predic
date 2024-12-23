import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# Fetching historical stock data
def fetch_stock_data(ticker, start_date, end_date):
    data = yf.download(ticker, start=start_date, end=end_date)
    data = data[['Close']]  # Use the closing price
    return data

# Feature engineering
def create_features(data):
    data['SMA_20'] = data['Close'].rolling(window=20).mean()  # 20-day Simple Moving Average
    data['SMA_50'] = data['Close'].rolling(window=50).mean()  # 50-day Simple Moving Average
    data['Return'] = data['Close'].pct_change()  # Daily returns
    data.dropna(inplace=True)  # Drop NaN values
    return data

# Preparing training and testing data
def prepare_data(data):
    X = data[['SMA_20', 'SMA_50', 'Return']]
    y = data['Close'].values.ravel()  # Ensure y is a 1D array
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test

# Training the model
def train_model(X_train, y_train):
    model = RandomForestRegressor(n_estimators=50, random_state=42)
    model.fit(X_train, y_train)
    return model

# Plotting predictions vs actual values
def plot_predictions(y_test, y_pred):
    plt.figure(figsize=(10, 6))
    plt.plot(y_test, label='Actual Prices', alpha=0.8)
    plt.plot(y_pred, label='Predicted Prices', alpha=0.8)
    plt.legend()
    plt.xlabel("Time")
    plt.ylabel("Stock Price")
    plt.title("Stock Price Prediction")
    plt.show()

# Main function
if __name__ == "__main__":
    ticker = 'HFG'  # Replace with your preferred stock ticker
    start_date = '2017-01-01'
    end_date = '2024-12-01'

    # Fetch and preprocess data
    data = fetch_stock_data(ticker, start_date, end_date)
    data = create_features(data)


    # Split data
    X_train, X_test, y_train, y_test = prepare_data(data)

    # Train model
    model = train_model(X_train, y_train)

    # Predictions
    y_pred = model.predict(X_test)

    # Evaluate model
    mse = mean_squared_error(y_test, y_pred)
    print(f"Mean Squared Error: {mse}")

    # Plot predictions
    plot_predictions(y_test, y_pred)
