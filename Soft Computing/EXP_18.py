from sklearn.datasets import fetch_lfw_people
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import numpy as np
import random
lfw = fetch_lfw_people(min_faces_per_person=20, resize=0.4)
X = lfw.data  # Flattened pixel data
y = lfw.target  # Labels (person IDs)
target_names = lfw.target_names
fig, axes = plt.subplots(2, 5, figsize=(12, 6))
for i, ax in enumerate(axes.ravel()):
    ax.imshow(lfw.images[i], cmap='gray')
    ax.set_title(target_names[y[i]])
    ax.axis('off')
plt.tight_layout()
plt.show()
encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y)
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.3, random_state=42)
input_size = X_train.shape[1]
hidden_size = 100 
output_size = len(target_names)
population_size = 10
num_generations = 20
mutation_rate = 0.1
def sigmoid(x):
    return 1 / (1 + np.exp(-x))
def fitness(weights, X_train, y_train):
    input_hidden = weights[:input_size * hidden_size].reshape(input_size, hidden_size)
    hidden_output = weights[input_size * hidden_size:].reshape(hidden_size, output_size)
    hidden_layer = sigmoid(np.dot(X_train, input_hidden))
    output_layer = sigmoid(np.dot(hidden_layer, hidden_output))
    predictions = np.argmax(output_layer, axis=1)
    return accuracy_score(y_train, predictions)
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
def genetic_algorithm(X_train, y_train):
    population = initialize_population()
    best_fitness_over_generations = []
    for generation in range(num_generations):
        fitness_scores = [fitness(ind, X_train, y_train) for ind in population]
        best_individual = population[np.argmax(fitness_scores)]
        best_fitness = max(fitness_scores)
        best_fitness_over_generations.append(best_fitness)
        print(f"Generation {generation+1} - Best Fitness: {best_fitness:.4f}")
        new_population = []
        for _ in range(population_size // 2):
            parent1, parent2 = select_parents(population, fitness_scores)
            child1, child2 = crossover(parent1, parent2)
            new_population.extend([mutate(child1), mutate(child2)])
        population = new_population
    return best_individual, best_fitness_over_generations
best_weights, fitness_history = genetic_algorithm(X_train, y_train)
plt.plot(fitness_history, marker='o')
plt.title('Fitness over Generations')
plt.xlabel('Generation')
plt.ylabel('Fitness (Accuracy)')
plt.grid()
plt.show()

# Test Optimized Weights
input_hidden = best_weights[:input_size * hidden_size].reshape(input_size, hidden_size)
hidden_output = best_weights[input_size * hidden_size:].reshape(hidden_size, output_size)

hidden_layer = sigmoid(np.dot(X_test, input_hidden))
output_layer = sigmoid(np.dot(hidden_layer, hidden_output))

predictions = np.argmax(output_layer, axis=1)
test_accuracy = accuracy_score(y_test, predictions)

print(f"Test Accuracy: {test_accuracy:.4f}")
