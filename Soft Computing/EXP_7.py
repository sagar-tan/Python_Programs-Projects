import numpy as np
# Define the detection levels for two sensors (as fuzzy sets)
sensor_1 = np.array([0.3, 0.7, 1.0, 0.5])
sensor_2 = np.array([0.5, 0.6, 0.9, 0.7])
# Fuzzy Union (max)
fuzzy_union = np.maximum(sensor_1, sensor_2)
# Fuzzy Intersection (min)
fuzzy_intersection = np.minimum(sensor_1, sensor_2)
# Fuzzy Difference (A - B)
fuzzy_difference = np.maximum(sensor_1 - sensor_2, 0)
# Fuzzy Complement of Sensor 1
fuzzy_complement_1 = 1 - sensor_1
# Algebraic Sum
fuzzy_algebraic_sum = sensor_1 + sensor_2 - sensor_1 * sensor_2
# Algebraic Product
fuzzy_algebraic_product = sensor_1 * sensor_2
# Bounded Sum
fuzzy_bounded_sum = np.minimum(1, sensor_1 + sensor_2)
# Bounded Difference
fuzzy_bounded_difference = np.maximum(0, sensor_1 - sensor_2)
# Drastic Sum
def drastic_sum(a, b):
    return np.where(a == 0, b, np.where(b == 0, a, 1))
fuzzy_drastic_sum = drastic_sum(sensor_1, sensor_2)
# Drastic Product
def drastic_product(a, b):
    return np.where((a == 0) | (b == 0), 0, np.minimum(a, b))
fuzzy_drastic_product = drastic_product(sensor_1, sensor_2)
# Print all results
print("Sensor 1 Detection Levels:", sensor_1)
print("Sensor 2 Detection Levels:", sensor_2)
print("Union (max):", fuzzy_union)
print("Intersection (min):", fuzzy_intersection)
print("Difference (Sensor 1 - Sensor 2):", fuzzy_difference)
print("Complement of Sensor 1:", fuzzy_complement_1)
print("Algebraic Sum:", fuzzy_algebraic_sum)
print("Algebraic Product:", fuzzy_algebraic_product)
print("Bounded Sum:", fuzzy_bounded_sum)
print("Bounded Difference:", fuzzy_bounded_difference)
print("Drastic Sum:", fuzzy_drastic_sum)
print("Drastic Product:", fuzzy_drastic_product)
