#Lambda-Cuts and Operations on Fuzzy Sets: Union, Intersection, and Complement
import numpy as np
# Defining the fuzzy sets A and B
A = np.array([0.2, 0.3, 0.4, 0.7, 0.1])
B = np.array([0.4, 0.5, 0.6, 0.8, 0.9])
# Defining the λ-cuts
lambdas = [0.7, 0.2, 0.6, 0.5, 0.7, 0.3, 0.6, 0.8]
# Union of A and B: max(A, B)
A_union_B = np.maximum(A, B)
# Intersection of A and B: min(A, B)
A_intersect_B = np.minimum(A, B)
# Applying λ-cuts
def lambda_cut(fuzzy_set, lambda_value):
    return [xi if mu >= lambda_value else 0 for xi, mu in enumerate(fuzzy_set, 1)]
# Compute λ-cuts for each case
print("Lambda cuts and operations:")
for lam in lambdas:
    A_cut = lambda_cut(A, lam)
    B_cut = lambda_cut(B, lam)
    print(f"\nλ-cut for λ = {lam}:")
    print(f"A: {A_cut}")
    print(f"B: {B_cut}")
    # Union and Intersection with λ
    A_union_B_cut = lambda_cut(A_union_B, lam)
    A_intersect_B_cut = lambda_cut(A_intersect_B, lam)
    print(f"A ∪ B: {A_union_B_cut}")
    print(f"A ∩ B: {A_intersect_B_cut}")
