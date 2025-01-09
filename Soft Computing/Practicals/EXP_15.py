#Implementing Fuzzy Logic-Based Traffic Light Control System.

import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl
vehicle_density = ctrl.Antecedent(np.arange(0, 11, 1), 'vehicle_density')
pedestrian_presence = ctrl.Antecedent(np.arange(0, 11, 1), 'pedestrian_presence')
time_of_day = ctrl.Antecedent(np.arange(0, 25, 1), 'time_of_day')
light_duration = ctrl.Consequent(np.arange(0, 61, 1), 'light_duration')
vehicle_density['low'] = fuzz.trimf(vehicle_density.universe, [0, 0, 5])
vehicle_density['medium'] = fuzz.trimf(vehicle_density.universe, [0, 5, 10])
vehicle_density['high'] = fuzz.trimf(vehicle_density.universe, [5, 10, 10])
pedestrian_presence['none'] = fuzz.trimf(pedestrian_presence.universe, [0, 0, 5])
pedestrian_presence['some'] = fuzz.trimf(pedestrian_presence.universe, [0, 5, 10])
pedestrian_presence['high'] = fuzz.trimf(pedestrian_presence.universe, [5, 10, 10])
time_of_day['morning'] = fuzz.trimf(time_of_day.universe, [0, 0, 12])
time_of_day['afternoon'] = fuzz.trimf(time_of_day.universe, [0, 12, 24])
time_of_day['night'] = fuzz.trimf(time_of_day.universe, [12, 24, 24])
light_duration['short'] = fuzz.trimf(light_duration.universe, [0, 0, 30])
light_duration['medium'] = fuzz.trimf(light_duration.universe, [0, 30, 60])
light_duration['long'] = fuzz.trimf(light_duration.universe, [30, 60, 60])
rule1 = ctrl.Rule(vehicle_density['low'] &pedestrian_presence['none'] &time_of_day['morning'], light_duration['long'])
rule2 = ctrl.Rule(vehicle_density['medium'] &pedestrian_presence['none'] &time_of_day['morning'], light_duration['medium'])
rule3 = ctrl.Rule(vehicle_density['high'] &pedestrian_presence['none'] &time_of_day['morning'], light_duration['short'])
traffic_ctrl = ctrl.ControlSystem([rule1, rule2, rule3])
traffic_light = ctrl.ControlSystemSimulation(traffic_ctrl)
traffic_light.input['vehicle_density'] = 3
traffic_light.input['pedestrian_presence'] = 0
traffic_light.input['time_of_day'] = 10
traffic_light.compute()
print("Recommended Traffic Light Duration:", traffic_light.output['light_duration'])
vehicle_density.view()
pedestrian_presence.view()
time_of_day.view()
light_duration.view()
