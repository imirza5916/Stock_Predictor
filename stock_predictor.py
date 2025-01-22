import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb

# Function to fetch stock data
def get_stock_data(ticker):
    stock = yf.download(ticker, period="5y")  # Fetch last 5 years of data
    stock["50_MA"] = stock["Close"].rolling(window=50).mean()
    stock["200_MA"] = stock["Close"].rolling(window=200).mean()

    # Relative Strength Index (RSI)
    delta = stock["Close"].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    stock["RSI"] = 100 - (100 / (1 + rs))

    # Moving Average Convergence Divergence (MACD)
    short_ema = stock["Close"].ewm(span=12, adjust=False).mean()
    long_ema = stock["Close"].ewm(span=26, adjust=False).mean()
    stock["MACD"] = short_ema - long_ema

    # Bollinger Bands
    stock["BB_Upper"] = stock["Close"].rolling(window=20).mean() + (2 * stock["Close"].rolling(window=20).std())
    stock["BB_Lower"] = stock["Close"].rolling(window=20).mean() - (2 * stock["Close"].rolling(window=20).std())

    # Drop rows with NaN values
    stock.dropna(inplace=True)

    return stock

# Train the model
def train_model(stock):
    X = stock[["Close", "50_MA", "200_MA", "RSI", "MACD", "BB_Upper", "BB_Lower", "Volume"]]
    y = stock["Close"].shift(-1)  # Predict next day's close

    # Ensure X and y have the same number of rows
    X, y = X.iloc[:-1], y.dropna()  # Drop last row from X to match y length

    # Split into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Convert `y_train` to 1D array
    y_train = y_train.values.ravel()

    # Scale data
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Use both RandomForest and XGBoost
    rf_model = RandomForestRegressor(n_estimators=200, random_state=42)
    rf_model.fit(X_train, y_train)

    xgb_model = xgb.XGBRegressor(objective="reg:squarederror", n_estimators=200)
    xgb_model.fit(X_train, y_train)

    return rf_model, xgb_model, scaler, X_test, y_test

# Predict the next N days
def predict_future_prices(models, scaler, stock, future_days=5):
    latest_data = stock.iloc[-1][["Close", "50_MA", "200_MA", "RSI", "MACD", "BB_Upper", "BB_Lower", "Volume"]].values.reshape(1, -1)
    latest_data_scaled = scaler.transform(latest_data)

    predictions = []
    for _ in range(future_days):
        rf_pred = models[0].predict(latest_data_scaled)[0]
        xgb_pred = models[1].predict(latest_data_scaled)[0]
        avg_pred = (rf_pred + xgb_pred) / 2  # Averaging both model predictions
        predictions.append(avg_pred)

        # Update latest data with new prediction
        latest_data[0][0] = avg_pred  # Update "Close" price
        latest_data_scaled = scaler.transform(latest_data)

    return predictions

# Plot stock data with future predictions
def plot_stock(stock, ticker, predicted_prices, future_dates=5):
    """
    Plot stock data with moving averages, Bollinger Bands, and future prediction.
    """
    plt.figure(figsize=(12, 6))

    # Historical stock price
    sns.lineplot(x=stock.index, y=stock["Close"].values.flatten(), label=f"{ticker} Closing Price", color="blue")

    # Moving Averages
    sns.lineplot(x=stock.index, y=stock["50_MA"].values.flatten(), label="50-day MA", color="red")
    sns.lineplot(x=stock.index, y=stock["200_MA"].values.flatten(), label="200-day MA", color="green")

    # Bollinger Bands
    plt.fill_between(stock.index, stock["BB_Upper"].values.flatten(), stock["BB_Lower"].values.flatten(), color='gray', alpha=0.2)

    # Future Predictions
    future_index = pd.date_range(start=stock.index[-1], periods=future_dates + 1, freq="B")[1:]
    plt.plot(future_index, predicted_prices, "ro--", label="Predicted Prices", markersize=5)

    # Labels and Title
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.title(f"{ticker} Stock Price with Moving Averages and Bollinger Bands")
    plt.legend()
    plt.grid()
    plt.show()

# Main function
def main():
    ticker = input("\nEnter stock ticker symbol (e.g., AAPL): ").upper()
    stock_data = get_stock_data(ticker)

    rf_model, xgb_model, scaler, X_test, y_test = train_model(stock_data)
    predicted_prices = predict_future_prices((rf_model, xgb_model), scaler, stock_data, future_days=5)

    print("\n📊 Model Evaluation:")
    y_pred_rf = rf_model.predict(X_test)
    y_pred_xgb = xgb_model.predict(X_test)
    y_pred = (y_pred_rf + y_pred_xgb) / 2  # Averaging predictions

    # Convert to NumPy arrays
    y_test = y_test.to_numpy().ravel()
    y_pred = y_pred.ravel()

    # Compute error metrics
    mae = np.abs(y_test - y_pred).mean()
    rmse = np.sqrt(((y_test - y_pred) ** 2).mean())

    # Display results
    print(f"🔹 Mean Absolute Error: {mae:.2f}")
    print(f"🔹 Root Mean Squared Error: {rmse:.2f}")
    print(f"📈 Predicted Next 5-Day Prices: {', '.join([f'${p:.2f}' for p in predicted_prices])}\n")

    # Plot stock data
    plot_stock(stock_data, ticker, predicted_prices)

if __name__ == "__main__":
    main()