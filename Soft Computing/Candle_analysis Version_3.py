import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM
import plotly.graph_objects as go
import os
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, precision_score, recall_score, f1_score


# Disable oneDNN optimizations for TensorFlow (optional for compatibility)
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# Step 1: Load and preprocess the data
file_path = r'C:\Users\tanwa\Python\EURUSD_Candlestick_5_M_BID_14.10.2024-19.10.2024 (1).csv'

# Load data
data = pd.read_csv(file_path)
data['Local time'] = pd.to_datetime(data['Local time'], format='%d.%m.%Y %H:%M:%S.%f GMT%z', errors='coerce')
data.dropna(inplace=True)

# Focus on relevant columns
ohlc_data = data[['Open', 'High', 'Low', 'Close']]

# Scale data for RNN
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(ohlc_data)

# Prepare sequences for training
sequence_length = 60
X, y = [], []
for i in range(sequence_length, len(scaled_data)):
    X.append(scaled_data[i-sequence_length:i])
    y.append(scaled_data[i, 3])  # Predict 'Close'

X, y = np.array(X), np.array(y)

# Step 2: Split Data into Training and Testing Sets
split_index = int(len(X) * 0.8)
X_train, X_test = X[:split_index], X[split_index:]
y_train, y_test = y[:split_index], y[split_index:]

# Step 3: Build and Train the RNN Model
model = Sequential([
    LSTM(100, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])),  # Increase LSTM units
    LSTM(100, return_sequences=False),
    Dense(50),  # Increase Dense units to capture more complexity
    Dense(1)
])

model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(X_train, y_train, epochs=50, batch_size=32, verbose=1)  # Increase epochs

# Step 4: Predict the Trend for the Next Few Days
# Use the last 60 data points from the dataset to predict the next 15 days for a stronger trend
last_sequence = scaled_data[-sequence_length:]
input_sequence = last_sequence.reshape(1, sequence_length, -1)

future_predictions = []
for _ in range(15):  # Predict for 15 days for a stronger trend
    predicted_scaled = model.predict(input_sequence)
    future_predictions.append(predicted_scaled[0, 0])
    # Update the input sequence with the new prediction
    next_input = np.concatenate((input_sequence[0][1:], [[predicted_scaled[0, 0]] * 4]), axis=0)
    input_sequence = next_input.reshape(1, sequence_length, -1)

# Rescale predictions to original values
future_predictions = scaler.inverse_transform(
    np.concatenate((np.zeros((15, 3)), np.array(future_predictions).reshape(-1, 1)), axis=1)
)[:, 3]

# Step 5: Plot Candle Chart and Prediction
fig = go.Figure()

# Add candlestick chart for the last week
week_data = data[data['Local time'] >= '2024-10-14']
fig.add_trace(go.Candlestick(
    x=week_data['Local time'],
    open=week_data['Open'],
    high=week_data['High'],
    low=week_data['Low'],
    close=week_data['Close'],
    name='Actual Data'
))

# Add prediction line chart
future_dates = pd.date_range(start=data['Local time'].iloc[-1], periods=16, freq='D')[1:]
fig.add_trace(go.Scatter(
    x=future_dates,
    y=future_predictions,
    mode='lines+markers',
    name='Predicted Trend',
    line=dict(color='blue', width=2)
))

# Add Moving Average (20-day moving average for stronger trend)
moving_average = data['Close'].rolling(window=20).mean().iloc[-len(week_data):]
fig.add_trace(go.Scatter(
    x=week_data['Local time'],
    y=moving_average,
    mode='lines',
    name='20-Day Moving Average',
    line=dict(color='orange', width=2, dash='dot')
))

# Update layout
fig.update_layout(
    title="EUR/USD Weekly Candlestick Chart with Stronger Predictions",
    xaxis_title="Date",
    yaxis_title="Price",
    template="plotly_dark",
    xaxis_rangeslider_visible=False
)

fig.show()

