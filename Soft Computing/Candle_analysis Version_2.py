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
n = 5 
df['min'] = df['Close'][(df['Close'].shift(n) > df['Close']) & (df['Close'].shift(-n) > df['Close'])]
df['max'] = df['Close'][(df['Close'].shift(n) < df['Close']) & (df['Close'].shift(-n) < df['Close'])]

support_levels = df[df['min'].notnull()]['Close'].values
resistance_levels = df[df['max'].notnull()]['Close'].values

high = df['Close'].max()
low = df['Close'].min()
fibonacci_levels = [0.236, 0.382, 0.5, 0.618, 0.786]
fib_levels = [high - (high - low) * level for level in fibonacci_levels]

scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(df['Close'].values.reshape(-1, 1))

prediction_days = 60
X_train, y_train = [], []

for x in range(prediction_days, len(scaled_data)):
    X_train.append(scaled_data[x-prediction_days:x, 0])
    y_train.append(scaled_data[x, 0])

X_train, y_train = np.array(X_train), np.array(y_train)
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
model.add(LSTM(units=50, return_sequences=False))
model.add(Dense(units=25))
model.add(Dense(units=1))

model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(X_train, y_train, batch_size=32, epochs=10)

X_test = scaled_data[-prediction_days:]
X_test = np.array([X_test]) 
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

future_predictions = []
for _ in range(288):  
    pred = model.predict(X_test)
    future_predictions.append(pred)

    pred = np.reshape(pred, (1, 1, 1))
    X_test = np.append(X_test[:, 1:, :], pred, axis=1)
future_predictions = scaler.inverse_transform(np.array(future_predictions).reshape(-1, 1))
future_dates = pd.date_range(df.index[-1], periods=len(future_predictions), freq='5T')

# Plot the results
plt.figure(figsize=(14, 8))
plt.plot(df['Close'], label='Existing Data', color='blue')

# Plot the predicted values
plt.plot(future_dates, future_predictions, color='green', label='Predicted Close Price')

# Plot Fibonacci and support/resistance levels
for fib in fib_levels:
    plt.axhline(fib, color='purple', linestyle='--', label=f'Fib level: {fib:.4f}')
plt.scatter(df.index, df['min'], color='red', label='Support Levels')
plt.scatter(df.index, df['max'], color='green', label='Resistance Levels')

plt.title('Price Prediction with Bullish Scenario')
plt.xlabel('Date')
plt.ylabel('Close Price')
plt.legend()
plt.show()
