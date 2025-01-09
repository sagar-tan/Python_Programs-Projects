#BACKPROPAGATION

import numpy as np
# Activation function: Sigmoid and its derivative
def sigmoid(x):
    return 1 / (1 + np.exp(-x))
def sigmoid_derivative(x):
    return x * (1 - x)
# Training data: AND function
inputs = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
])
# Target outputs for AND function
targets = np.array([[0], [0], [0], [1]])
# Initialize weights and biases
np.random.seed(1)  # For reproducibility
input_layer_neurons = 2    # Number of input neurons
hidden_layer_neurons = 2   # Number of hidden neurons
output_layer_neurons = 1   # Number of output neurons
# Weights and biases
weights_input_hidden = np.random.rand(input_layer_neurons, hidden_layer_neurons)
weights_hidden_output = np.random.rand(hidden_layer_neurons, output_layer_neurons)
bias_hidden = np.random.rand(1, hidden_layer_neurons)
bias_output = np.random.rand(1, output_layer_neurons)
# Learning rate
learning_rate = 0.1
epochs = 10000
# Training process
for epoch in range(epochs):
    # Forward pass
    hidden_input = np.dot(inputs, weights_input_hidden) + bias_hidden
    hidden_output = sigmoid(hidden_input)
    final_input = np.dot(hidden_output, weights_hidden_output) + bias_output
    final_output = sigmoid(final_input)
    # Calculate the error (loss)
    error = targets - final_output
    # Backpropagation
    d_output = error * sigmoid_derivative(final_output)
    # Error at hidden layer
    error_hidden = d_output.dot(weights_hidden_output.T)
    d_hidden = error_hidden * sigmoid_derivative(hidden_output)
    # Update weights and biases
    weights_hidden_output += hidden_output.T.dot(d_output) * learning_rate
    weights_input_hidden += inputs.T.dot(d_hidden) * learning_rate
    bias_output += np.sum(d_output, axis=0, keepdims=True) * learning_rate
    bias_hidden += np.sum(d_hidden, axis=0, keepdims=True) * learning_rate
# Display final outputs
print("Final outputs after training:")
print(final_output)
