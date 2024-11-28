import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from keras.models import Sequential # type: ignore
from keras.layers import Dense # type: ignore

temperature = ctrl.Antecedent(np.arange(0, 41, 1), 'temperature')
humidity = ctrl.Antecedent(np.arange(0, 101, 1), 'humidity')
fan_speed = ctrl.Consequent(np.arange(0, 31, 1), 'fan_speed')
temperature['low'] = fuzz.trimf(temperature.universe, [0, 0, 20])
temperature['medium'] = fuzz.trimf(temperature.universe, [10, 20, 30])
temperature['high'] = fuzz.trimf(temperature.universe, [20, 40, 40])
humidity['low'] = fuzz.trimf(humidity.universe, [0, 0, 50])
humidity['medium'] = fuzz.trimf(humidity.universe, [30, 50, 70])
humidity['high'] = fuzz.trimf(humidity.universe, [50, 100, 100])
fan_speed['low'] = fuzz.trimf(fan_speed.universe, [0, 0, 15])
fan_speed['medium'] = fuzz.trimf(fan_speed.universe, [10, 15, 20])
fan_speed['high'] = fuzz.trimf(fan_speed.universe, [15, 30, 30])
rule1 = ctrl.Rule(temperature['high'] & humidity['low'], fan_speed['high'])
rule2 = ctrl.Rule(temperature['medium'] & humidity['medium'], fan_speed['medium'])
rule3 = ctrl.Rule(temperature['low'] | humidity['high'], fan_speed['low'])
fan_control = ctrl.ControlSystem([rule1, rule2, rule3])
fan_simulation = ctrl.ControlSystemSimulation(fan_control)
data_size = 100
temperature_data = np.random.randint(0, 41, data_size)
humidity_data = np.random.randint(0, 101, data_size)
fan_speeds = []
for temp, hum in zip(temperature_data, humidity_data):
    fan_simulation.input['temperature'] = temp
    fan_simulation.input['humidity'] = hum
    try:
        fan_simulation.compute()
        fan_speeds.append(fan_simulation.output['fan_speed'])
    except KeyError:
        fan_speeds.append(0)  # Set a default value or handle appropriately

fan_speeds = np.array(fan_speeds)
X = np.vstack((temperature_data, humidity_data)).T
y = fan_speeds
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = Sequential()
model.add(Dense(10, input_dim=2, activation='relu'))  # Input layer for temperature and humidity
model.add(Dense(10, activation='relu'))               # Hidden layer
model.add(Dense(1, activation='linear'))              # Output layer for fan speed
model.compile(optimizer='adam', loss='mse')
model.fit(X_train, y_train, epochs=100, batch_size=10, verbose=1)
loss = model.evaluate(X_test, y_test)
print(f'Test Loss: {loss}')
predicted_fan_speeds = model.predict(X_test)
plt.scatter(y_test, predicted_fan_speeds)
plt.xlabel('True Fan Speed')
plt.ylabel('Predicted Fan Speed')
plt.title('True vs Predicted Fan Speeds')
plt.show()
