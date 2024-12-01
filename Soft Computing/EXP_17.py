#Creating a face recognition system with a genetic neuro-hybrid approach involves combining neural networks for feature extraction and a genetic algorithm for optimizing parameters (such as weights, thresholds, or architecture).

import numpy as np
import matplotlib.pyplot as plt
import random
X = np.array([
    [1, 1, 1],  # Face 1
    [1, 0, 1],  # Face 2
    [0, 1, 1],  # Face 3
    [0, 0, 1],  # Face 4
])
y = np.array([0, 1, 2, 3])
input_size = 3
hidden_size = 4
output_size = 4
population_size = 10
num_generations = 20
mutation_rate = 0.1
def sigmoid(x):
    return 1 / (1 + np.exp(-x))
def fitness(weights):
    input_hidden = weights[:input_size * hidden_size].reshape(input_size, hidden_size)
    hidden_output = weights[input_size * hidden_size:].reshape(hidden_size, output_size)
    hidden_layer = sigmoid(np.dot(X, input_hidden))
    output_layer = sigmoid(np.dot(hidden_layer, hidden_output))
    predictions = np.argmax(output_layer, axis=1)
    accuracy = np.mean(predictions == y)
    return accuracy
def initialize_population():
    return [np.random.uniform(-1, 1, input_size * hidden_size + hidden_size * output_size) for _ in range(population_size)]
def select_parents(population, fitness_scores):
    total_fitness = sum(fitness_scores)
    selection_probs = [score / total_fitness for score in fitness_scores]
    parents = random.choices(population, weights=selection_probs, k=2)
    return parents
def crossover(parent1, parent2):
    point = random.randint(0, len(parent1) - 1)
    child1 = np.concatenate((parent1[:point], parent2[point:]))
    child2 = np.concatenate((parent2[:point], parent1[point:]))
    return child1, child2
def mutate(chromosome):
    for i in range(len(chromosome)):
        if random.random() < mutation_rate:
            chromosome[i] += np.random.uniform(-0.5, 0.5)
    return chromosome
def genetic_algorithm():
    population = initialize_population()
    for generation in range(num_generations):
        fitness_scores = [fitness(ind) for ind in population]
        best_individual = population[np.argmax(fitness_scores)]
        best_fitness = max(fitness_scores)
        print(f"Generation {generation+1} - Best Fitness: {best_fitness:.4f}")
        new_population = []
        for _ in range(population_size // 2):
            parent1, parent2 = select_parents(population, fitness_scores)
            child1, child2 = crossover(parent1, parent2)
            new_population.extend([mutate(child1), mutate(child2)])
        population = new_population
    return best_individual
best_weights = genetic_algorithm()
input_hidden = best_weights[:input_size * hidden_size].reshape(input_size, hidden_size)
hidden_output = best_weights[input_size * hidden_size:].reshape(hidden_size, output_size)
hidden_layer = sigmoid(np.dot(X, input_hidden))
output_layer = sigmoid(np.dot(hidden_layer, hidden_output))
print("\nPredictions:")
predictions = np.argmax(output_layer, axis=1)
print("Predicted Labels:", predictions)
print("Actual Labels:   ", y)
