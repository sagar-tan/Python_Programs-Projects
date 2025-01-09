#Generate AND/OR function using McCulloch-Pitts neural net.
def mcculloch_pitts_neuron(inputs, weights, threshold):
    # Calculate the weighted sum
    weighted_sum = sum(i * w for i, w in zip(inputs, weights))
    # Apply the activation function
    output = 1 if weighted_sum >= threshold else 0
    return output
""""
# AND function
inputs = [1, 1]  # Example input
weights = [1, 1]
threshold = 2

# OR function
threshold = 1

"""
# NOT function
inputs = [1]  # Example input
weights = [-1]
threshold = 0


output = mcculloch_pitts_neuron(inputs, weights, threshold)
print(f"Output of NOT function: {output}")
