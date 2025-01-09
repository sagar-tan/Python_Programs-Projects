#Max-Min and Max-Product Composition of Fuzzy Relations To compute and compare the Max-Min and Max-Product compositions of two given fuzzy relations, demonstrating different methods of composing fuzzy relations for decision-making in fuzzy logic systems.
import numpy as np
# Define fuzzy relations R and S
R = np.array([[0.6, 0.3],
              [0.2, 0.9]])
S = np.array([[1, 0.5, 0.3],
              [0.8, 0.4, 0.7]])
# Initialize the result matrices for Max-Min and Max-Product
T_max_min = np.zeros((R.shape[0], S.shape[1]))
T_max_product = np.zeros((R.shape[0], S.shape[1]))
# Perform both Max-Min and Max-Product compositions
for i in range(R.shape[0]):  # For each row in R
    for j in range(S.shape[1]):  # For each column in S
        min_values = []
        product_values = []
        for k in range(R.shape[1]):  # Iterate over the second dimension (shared by R and S)
            min_values.append(min(R[i, k], S[k, j]))  # Calculate the min for Max-Min composition
            product_values.append(R[i, k] * S[k, j])  # Calculate the product for Max-Product composition
        T_max_min[i, j] = max(min_values)  # Store the max of the min values for Max-Min composition
        T_max_product[i, j] = max(product_values)  # Store the max of the product values for Max-Product composition
# Output the resulting composition matrices
print("Max-Min Composition:")
print(T_max_min)
print("\nMax-Product Composition:")
print(T_max_product)
