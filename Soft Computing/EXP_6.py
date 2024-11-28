#To explore and implement fundamental fuzzy set operations (union, intersection, difference, complement, etc.) on given sensor detection levels.
import numpy as np
# Define the detection levels for each sensor at different gain settings
sensor_1 = np.array([0.3, 0.7, 1.0, 0.5])
sensor_2 = np.array([0.5, 0.6, 0.9, 0.7])
# Fuzzy union: Maximum detection level between the two sensors
fuzzy_union = np.maximum(sensor_1, sensor_2)
# Fuzzy intersection: Minimum detection level between the two sensors
fuzzy_intersection = np.minimum(sensor_1, sensor_2)
# Fuzzy difference: Sensor 1 - Sensor 2
fuzzy_difference = np.maximum(sensor_1 - sensor_2, 0)
# Print the results
print("Sensor 1 Detection Levels:", sensor_1)
print("Sensor 2 Detection Levels:", sensor_2)
print("Union (max) of Detection Levels:", fuzzy_union)
print("Intersection (min) of Detection Levels:", fuzzy_intersection)
print("Difference (Sensor 1 - Sensor 2):", fuzzy_difference)
