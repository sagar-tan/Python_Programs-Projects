#Implement AND function using perceptron networks for bipolar inputs and targets.
import numpy as np
# Activation function for the perceptron
def activation(net_input):
    return 1 if net_input >= 0 else -1 
# Perceptron learning algorithm
def perceptron_train(inputs, targets, learning_rate=0.1, epochs=100):
    # Initialize weights and bias randomly
    weights = np.random.randn(inputs.shape[1])
    bias = np.random.randn()

    for epoch in range(epochs):
        total_error = 0
        for x, target in zip(inputs, targets):
            # Calculate the net input
            net_input = np.dot(x, weights) + bias
            # Predict the output
            output = activation(net_input)
            # Calculate the error (difference between target and predicted output)
            error = target - output
            # Update weights and bias using the perceptron rule
            weights += learning_rate * error * x
            bias += learning_rate * error
            # Accumulate the total error
            total_error += abs(error)

        # Print the total error for each epoch
        print(f'Epoch {epoch + 1}/{epochs}, Total Error: {total_error}')
        # If total error is 0, we can stop early (the network has learned)
        if total_error == 0:
            break

    return weights, bias

# Test the perceptron network
def perceptron_predict(inputs, weights, bias):
    # Predict outputs for all inputs
    predictions = []
    for x in inputs:
        net_input = np.dot(x, weights) + bias
        output = activation(net_input)
        predictions.append(output)
    return np.array(predictions)

# AND function using bipolar inputs (-1, 1)
inputs = np.array([
    [-1, -1],
    [-1,  1],
    [ 1, -1],
    [ 1,  1]
])

# AND function outputs (bipolar targets)
targets = np.array([-1, -1, -1, 1])

# Train the Perceptron network
learning_rate = 0.1
epochs = 100
weights, bias = perceptron_train(inputs, targets, learning_rate, epochs)

# Test the network on the same inputs
predictions = perceptron_predict(inputs, weights, bias)

# Display results
print("\nFinal Weights:", weights)
print("Final Bias:", bias)
print("\nPredictions:")
for i, (x, target) in enumerate(zip(inputs, targets)):
    print(f"Input: {x}, Target: {target}, Predicted: {predictions[i]}")
