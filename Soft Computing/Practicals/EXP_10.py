import numpy as np
fuzzy_A = {'x1': 0.2, 'x2': 0.3, 'x3': 0.4, 'x4': 0.7, 'x5': 0.1}
fuzzy_B = {'x1': 0.4, 'x2': 0.5, 'x3': 0.6, 'x4': 0.8, 'x5': 0.9}
lambda_cut = 0.7
lambda_cut_A = {x: fuzzy_A[x] for x in fuzzy_A if fuzzy_A[x] >= lambda_cut}
print(f"Lambda-cut of A (lambda={lambda_cut}): {lambda_cut_A}")
lambda_cut_B = {x: fuzzy_B[x] for x in fuzzy_B if fuzzy_B[x] >= lambda_cut}
print(f"Lambda-cut of B (lambda={lambda_cut}): {lambda_cut_B}")
union_AB = {x: max(fuzzy_A[x], fuzzy_B[x]) for x in fuzzy_A if max(fuzzy_A[x], fuzzy_B[x]) >= lambda_cut}
print(f"Union of A and B (lambda={lambda_cut}): {union_AB}")
intersection_AB = {x: min(fuzzy_A[x], fuzzy_B[x]) for x in fuzzy_A if min(fuzzy_A[x], fuzzy_B[x]) >= lambda_cut}
print(f"Intersection of A and B (lambda={lambda_cut}): {intersection_AB}")
