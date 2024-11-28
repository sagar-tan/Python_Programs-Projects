# Write a program to perform Union, Intersection and Complement operations.

import numpy as np
# Enter Data
u = np.array(eval(input('Enter First Matrix (as a list of lists, e.g., [[1,2],[3,4]]): ')), dtype=float)
v = np.array(eval(input('Enter Second Matrix (as a list of lists, e.g., [[1,0],[0,1]]): ')), dtype=float)
# To Perform Operations
w = np.maximum(u, v)  # Union
p = np.minimum(u, v)  # Intersection
q1 = 1 - u            # Complement of the first matrix
q2 = 1 - v            # Complement of the second matrix
# Display Output
print('Union Of Two Matrices')
print(w)
print('Intersection Of Two Matrices')
print(p)
print('Complement Of First Matrix')
print(q1)
print('Complement Of Second Matrix')
print(q2)
