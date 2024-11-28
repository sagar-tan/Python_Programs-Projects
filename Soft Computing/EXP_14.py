#Use a genetic algorithm to find the maximum number of 1's in a binary string of fixed length (also known as the "One-Max Problem").

import numpy as np

# Parameters
population_size = 10
chromosome_length = 8
generations = 50
p_c = 0.7  # Crossover probability
p_m = 0.01 # Mutation probability

# Generate Initial Population
def initialize_population(size, length):
    return np.random.randint(2, size=(size, length))

# Fitness Function
def fitness(chromosome):
    return np.sum(chromosome)  # Count of 1's

# Selection using Roulette Wheel
def roulette_wheel_selection(population, fitness_values):
    total_fitness = np.sum(fitness_values)
    selection_probs = fitness_values / total_fitness
    selected_idx = np.random.choice(len(population), p=selection_probs)
    return population[selected_idx]

# Crossover
def crossover(parent1, parent2, p_c):
    if np.random.rand() < p_c:
        point = np.random.randint(1, len(parent1) - 1)
        child1 = np.concatenate((parent1[:point], parent2[point:]))
        child2 = np.concatenate((parent2[:point], parent1[point:]))
        return child1, child2
    return parent1, parent2

# Mutation
def mutate(chromosome, p_m):
    for i in range(len(chromosome)):
        if np.random.rand() < p_m:
            chromosome[i] = 1 - chromosome[i]  # Flip bit
    return chromosome

# Genetic Algorithm
population = initialize_population(population_size, chromosome_length)

for gen in range(generations):
    fitness_values = np.array([fitness(chrom) for chrom in population])

    # Display progress
    print(f"Generation {gen + 1}: Max Fitness = {np.max(fitness_values)}")

    new_population = []

    # Create new population
    for _ in range(population_size // 2):
        parent1 = roulette_wheel_selection(population, fitness_values)
        parent2 = roulette_wheel_selection(population, fitness_values)

        # Crossover
        child1, child2 = crossover(parent1, parent2, p_c)

        # Mutation
        child1 = mutate(child1, p_m)
        child2 = mutate(child2, p_m)

        new_population.extend([child1, child2])

    population = np.array(new_population)

# Final Results
final_fitness = np.array([fitness(chrom) for chrom in population])
print("Final population fitness:", final_fitness)
print("Best solution:", population[np.argmax(final_fitness)])
