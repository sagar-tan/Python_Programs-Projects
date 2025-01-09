#Implementation of Mamdani-Type Fuzzy Inference System for Heater Power Control.

import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl
temperature = ctrl.Antecedent(np.arange(0, 101, 1), 'temperature')
heater_power = ctrl.Consequent(np.arange(0, 101, 1), 'heater_power')
temperature['cold'] = fuzz.trimf(temperature.universe, [0, 0, 50])
temperature['warm'] = fuzz.trimf(temperature.universe, [0, 50, 100])
temperature['hot'] = fuzz.trimf(temperature.universe, [50, 100, 100])
heater_power['low'] = fuzz.trimf(heater_power.universe, [0, 0, 50])
heater_power['medium'] = fuzz.trimf(heater_power.universe, [0, 50, 100])
heater_power['high'] = fuzz.trimf(heater_power.universe, [50, 100, 100])
rule1 = ctrl.Rule(temperature['cold'], heater_power['high'])
rule2 = ctrl.Rule(temperature['warm'], heater_power['medium'])
rule3 = ctrl.Rule(temperature['hot'], heater_power['low'])
temperature_ctrl = ctrl.ControlSystem([rule1, rule2, rule3])
heater_power_ctrl = ctrl.ControlSystemSimulation(temperature_ctrl)
heater_power_ctrl.input['temperature'] = 75
heater_power_ctrl.compute()
print("Heater Power:", heater_power_ctrl.output['heater_power'])