#Implementation of Fuzzy Logic-Based Tip Recommendation System for Dining Experience. Use Fuzzy toolbox to model tip value that is given after a dinner which can be-not good,satisfying,good and delightful and service which is poor,average or good and the tip value will range from Rs. 10 to 100.

import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl
satisfaction = ctrl.Antecedent(np.arange(0, 11, 1), 'satisfaction')
service = ctrl.Antecedent(np.arange(0, 11, 1), 'service')
tip = ctrl.Consequent(np.arange(10, 101, 1), 'tip')
satisfaction['not_good'] = fuzz.trimf(satisfaction.universe, [0, 0, 5])
satisfaction['satisfying'] = fuzz.trimf(satisfaction.universe, [0, 5, 10])
satisfaction['good'] = fuzz.trimf(satisfaction.universe, [5, 10, 10])
satisfaction['delightful'] = fuzz.trimf(satisfaction.universe, [10, 10, 10])
service['poor'] = fuzz.trimf(service.universe, [0, 0, 5])
service['average'] = fuzz.trimf(service.universe, [0, 5, 10])
service['good'] = fuzz.trimf(service.universe, [5, 10, 10])
tip['low'] = fuzz.trimf(tip.universe, [10, 10, 55])
tip['medium'] = fuzz.trimf(tip.universe, [10, 55, 100])
tip['high'] = fuzz.trimf(tip.universe, [55, 100, 100])
rule1 = ctrl.Rule(satisfaction['not_good'] | service['poor'], tip['low'])
rule2 = ctrl.Rule(satisfaction['satisfying'] & service['average'], tip['medium'])
rule3 = ctrl.Rule(satisfaction['good'] | service['good'], tip['high'])
rule4 = ctrl.Rule(satisfaction['delightful'] & service['good'], tip['high'])
tip_ctrl = ctrl.ControlSystem([rule1, rule2, rule3, rule4])
tip_sim = ctrl.ControlSystemSimulation(tip_ctrl)
tip_sim.input['satisfaction'] = 7
tip_sim.input['service'] = 8
tip_sim.compute()
print("Recommended tip value:", tip_sim.output['tip'])
tip.view(sim=tip_sim)   
