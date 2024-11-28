import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential  # type: ignore
from tensorflow.keras.layers import Dense, LSTM  # type: ignore
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

file_path = 'C:/Users/tanwa/Python/EURUSD_Candlestick_5_M_BID_14.10.2024-19.10.2024 (1).csv'
df = pd.read_csv(file_path)
df['Datetime'] = pd.to_datetime(df['Local time'], format='%d.%m.%Y %H:%M:%S.%f GMT%z')
df = df[['Datetime', 'Close']]
df.set_index('Datetime', inplace=True)

# Detect support and resistance levels using local minima and maxima
n = 5  # Number of points to consider for local minima/maxima
df['min'] = df['Close'][(df['Close'].shift(n) > df['Close']) & (df['Close'].shift(-n) > df['Close'])]
df['max'] = df['Close'][(df['Close'].shift(n) < df['Close']) & (df['Close'].shift(-n) < df['Close'])]

# Extract support and resistance levels
support_levels = df[df['min'].notnull()]['Close'].values
resistance_levels = df[df['max'].notnull()]['Close'].values

# Fibonacci levels calculation
high = df['Close'].max()
low = df['Close'].min()
fibonacci_levels = [0.236, 0.382, 0.5, 0.618, 0.786]
fib_levels = [high - (high - low) * level for level in fibonacci_levels]

# Scale data for LSTM model
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(df['Close'].values.reshape(-1, 1))

# Prepare training data (using the past 60 time steps to predict the next one)
prediction_days = 60
X_train, y_train = [], []

for x in range(prediction_days, len(scaled_data)):
    X_train.append(scaled_data[x-prediction_days:x, 0])
    y_train.append(scaled_data[x, 0])

X_train, y_train = np.array(X_train), np.array(y_train)
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

# Build LSTM model
model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
model.add(LSTM(units=50, return_sequences=False))
model.add(Dense(units=25))
model.add(Dense(units=1))

# Compile and train the model
model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(X_train, y_train, batch_size=32, epochs=10)

# Predict next values based on the last 60 data points
X_test = scaled_data[-prediction_days:]  # Get the last 60 scaled values
X_test = np.array([X_test])  # Reshape for prediction
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

# Generate predictions for 288 steps (one day)
future_predictions = []
for _ in range(288):  # 1 day of 5-minute candles
    pred = model.predict(X_test)
    future_predictions.append(pred[0])  # Append the first dimension of the prediction
    
    # Reshape 'pred' to have the shape (batch_size, 1, features)
    pred = np.reshape(pred, (1, 1, 1))
    
    # Now append 'pred' to 'X_test[:, 1:, :]'
    X_test = np.append(X_test[:, 1:, :], pred, axis=1)

# Inverse transform to get actual price predictions
future_predictions = scaler.inverse_transform(np.array(future_predictions).reshape(-1, 1))

# Simulate price action with more significant adjustments
def simulate_price_action(future_predictions, fib_levels, support_levels, resistance_levels, scenario):
    """
    Simulates price action based on given scenario and the influence of levels.
    """
    adjusted_predictions = []
    for pred in future_predictions:
        price = pred  # Directly use pred as it is already the price
        
        # Apply scenario biases:
        if scenario == "bullish":
            price *= 1.02  # Slight upward adjustment
        elif scenario == "bearish":
            price *= 0.98  # Slight downward adjustment

        # Stronger adjustments for Fibonacci levels
        for fib_level in fib_levels:
            if abs(price - fib_level) < 0.01 * price:  # Near Fibonacci level
                if scenario == "bullish":
                    price += (price * 0.02)  # Stronger upward breakout
                elif scenario == "bearish":
                    price -= (price * 0.02)  # Stronger downward rejection

        # Support/Resistance Bounce with stronger influence
        if scenario == "bullish":
            for resistance in resistance_levels:
                if abs(price - resistance) < 0.01 * price:  # Close to resistance
                    price -= price * 0.03  # Stronger rejection at resistance
                    break  # Exit loop after first hit

            for support in support_levels:
                if abs(price - support) < 0.01 * price:  # Close to support
                    price += price * 0.03  # Stronger bounce at support
                    break  # Exit loop after first hit

        adjusted_predictions.append(price)
    
    return adjusted_predictions

# Generate predictions for different scenarios
bullish_predictions = simulate_price_action(future_predictions, fib_levels, support_levels, resistance_levels, "bullish")
bearish_predictions = simulate_price_action(future_predictions, fib_levels, support_levels, resistance_levels, "bearish")
consolidation_predictions = simulate_price_action(future_predictions, fib_levels, support_levels, resistance_levels, "consolidation")

# Create a datetime range for future predictions
future_dates = pd.date_range(df.index[-1], periods=len(bullish_predictions), freq='5T')

# Plot the results
plt.figure(figsize=(14, 8))
plt.plot(df['Close'], label='Existing Data', color='blue')

# Plot the final adjusted predictions based on the bullish scenario
plt.plot(future_dates, bullish_predictions, color='green', label='Adjusted Bullish Prediction')

# Plot Fibonacci and support/resistance levels
for fib in fib_levels:
    plt.axhline(fib, color='purple', linestyle='--', label=f'Fib level: {fib:.4f}')
plt.scatter(df.index, df['min'], color='red', label='Support Levels')
plt.scatter(df.index, df['max'], color='green', label='Resistance Levels')

plt.title('Price Prediction with Enhanced Influence of Levels')
plt.xlabel('Date')
plt.ylabel('Close Price')
plt.legend()
plt.show() 
plt.show()
