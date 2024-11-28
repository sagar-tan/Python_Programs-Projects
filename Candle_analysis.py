import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
file_path = 'C:/Users/tanwa/Python/EURUSD_Candlestick_5_M_BID_14.10.2024-19.10.2024 (1).csv'
df = pd.read_csv(file_path)
df['Datetime'] = pd.to_datetime(df['Local time'], format='%d.%m.%Y %H:%M:%S.%f GMT%z')
df.set_index('Datetime', inplace=True)
df = df[['Close']]
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(df)
def prepare_data(data, time_step=60):
    X, y = [], []
    for i in range(len(data) - time_step - 1):
        X.append(data[i:(i + time_step), 0])
        y.append(data[i + time_step, 0])
    return np.array(X), np.array(y)
time_step = 60
X, y = prepare_data(scaled_data, time_step)
X = X.reshape(X.shape[0], X.shape[1], 1)
model = Sequential()
model.add(LSTM(50, return_sequences=True, input_shape=(time_step, 1)))
model.add(LSTM(50, return_sequences=False))
model.add(Dense(25))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(X, y, batch_size=64, epochs=10)
past_60_mins = scaled_data[-time_step:]
X_test = np.array([past_60_mins[:, 0]])
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)
predictions = []
for i in range(24):
    pred = model.predict(X_test)
    predictions.append(pred[0, 0])
    pred = np.reshape(pred, (1, 1, 1)) 
    X_test = np.append(X_test[:, 1:, :], pred, axis=1)
predictions = scaler.inverse_transform(np.array(predictions).reshape(-1, 1))

future_datetimes = pd.date_range(df.index[-1], periods=25, freq='5T')[1:]
plt.figure(figsize=(12, 6))
plt.plot(df.index[-60:], df['Close'][-60:], label='Historical Data (Last 60 intervals)')
plt.plot(future_datetimes, predictions, label='Predicted Data (Next 2 Hours)', color='red')
plt.xlabel('Datetime')
plt.ylabel('EUR/USD Price')
plt.title('Live Prediction Curve for EUR/USD (Next 2 Hours)')
plt.legend()
plt.show()
