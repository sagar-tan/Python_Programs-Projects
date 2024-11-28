'''
Using back-propagation network, find the new weights.
It is presented with the input pattern [0, 1] and the target output is 1.
Use a learning rate a =0.25 and binary sigmoidal activation function.
'''
import numpy as np
# Sigmoid activation function and its derivative
def sigmoid(x):
    return 1 / (1 + np.exp(-x))
def sigmoid_derivative(x):
    return x * (1 - x)
# Given data
inputs = np.array([[0, 1]])  # Input pattern [0, 1]
target = np.array([[1]])     # Target output 1
learning_rate = 0.25         # Learning rate alpha
# Initializing weights randomly
np.random.seed(42)  # For reproducibility
weights_input_hidden = np.random.rand(2, 2)  # 2 input neurons, 2 hidden neurons
weights_hidden_output = np.random.rand(2, 1)  # 2 hidden neurons, 1 output neuron
bias_hidden = np.random.rand(1, 2)  # Bias for hidden layer
bias_output = np.random.rand(1, 1)  # Bias for output layer
# Forward Pass
hidden_input = np.dot(inputs, weights_input_hidden) + bias_hidden
hidden_output = sigmoid(hidden_input)  # Activation function at hidden layer
final_input = np.dot(hidden_output, weights_hidden_output) + bias_output
final_output = sigmoid(final_input)  # Activation function at output layer
# Error Calculation
error = target - final_output
# Backward Pass
# Calculate the gradient for output layer
d_output = error * sigmoid_derivative(final_output)
# Calculate the gradient for hidden layer
error_hidden = d_output.dot(weights_hidden_output.T)
d_hidden = error_hidden * sigmoid_derivative(hidden_output)
# Update weights and biases
weights_hidden_output += hidden_output.T.dot(d_output) * learning_rate
weights_input_hidden += inputs.T.dot(d_hidden) * learning_rate
bias_output += np.sum(d_output, axis=0, keepdims=True) * learning_rate
bias_hidden += np.sum(d_hidden, axis=0, keepdims=True) * learning_rate
# Display results
print("Updated weights from input to hidden layer:\n", weights_input_hidden)
print("Updated weights from hidden to output layer:\n", weights_hidden_output)
print("Updated hidden layer bias:\n", bias_hidden)
print("Updated output layer bias:\n", bias_output)
