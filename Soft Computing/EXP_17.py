#knapsack problem

import random
import numpy as np

# Problem definition
items = [
    {"weight": 10, "value": 60},
    {"weight": 20, "value": 100},
    {"weight": 30, "value": 120}
]
max_weight = 50
population_size = 6
generations = 50
mutation_rate = 0.1

# Fitness function
def fitness(individual):
    total_weight = sum(ind["weight"] * gene for ind, gene in zip(items, individual))
    total_value = sum(ind["value"] * gene for ind, gene in zip(items, individual))
    return total_value if total_weight <= max_weight else 0

# Generate initial population
def generate_individual():
    return [random.randint(0, 1) for _ in range(len(items))]

def generate_population(size):
    return [generate_individual() for _ in range(size)]

# Selection
def selection(population):
    fitnesses = [fitness(ind) for ind in population]
    probabilities = [f / sum(fitnesses) for f in fitnesses]
    return population[np.random.choice(len(population), p=probabilities)]

# Crossover
def crossover(parent1, parent2):
    point = random.randint(1, len(items) - 1)
    child1 = parent1[:point] + parent2[point:]
    child2 = parent2[:point] + parent1[point:]
    return child1, child2

# Mutation
def mutate(individual):
    for i in range(len(individual)):
        if random.random() < mutation_rate:
            individual[i] = 1 - individual[i]
    return individual

# Genetic Algorithm
population = generate_population(population_size)
for gen in range(generations):
    new_population = []
    for _ in range(population_size // 2):
        parent1 = selection(population)
        parent2 = selection(population)
        child1, child2 = crossover(parent1, parent2)
        new_population.extend([mutate(child1), mutate(child2)])
    population = new_population

# Get the best solution
best_individual = max(population, key=fitness)
print("Best solution:", best_individual)
print("Best value:", fitness(best_individual))

