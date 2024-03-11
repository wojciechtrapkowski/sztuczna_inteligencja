from itertools import compress
import random
import time
import matplotlib.pyplot as plt

from data import *

def initial_population(individual_size, population_size):
    return [[random.choice([True, False]) for _ in range(individual_size)] for _ in range(population_size)]

def fitness(items, knapsack_max_capacity, individual):
    total_weight = sum(compress(items['Weight'], individual))
    if total_weight > knapsack_max_capacity:
        return 0
    return sum(compress(items['Value'], individual))

def population_best(items, knapsack_max_capacity, population):
    best_individual = None
    best_individual_fitness = -1
    for individual in population:
        individual_fitness = fitness(items, knapsack_max_capacity, individual)
        if individual_fitness > best_individual_fitness:
            best_individual = individual
            best_individual_fitness = individual_fitness
    return best_individual, best_individual_fitness


def generate_parents_and_elites(population):
    population_probability = []

    sum_of_probability = 0
    for individual in population:
        sum_of_probability += fitness(items, knapsack_max_capacity, individual)
    
    for individual in population:
        population_probability.append(fitness(items, knapsack_max_capacity, individual) / sum_of_probability)
    
    zipped_lists = list(zip(population, population_probability))

    sorted_lists = sorted(zipped_lists, key=lambda x: x[1], reverse=True)
    
    parents = random.choices(population, weights=population_probability, k=n_selection)
    elites = [individual[0] for individual in sorted_lists[:n_elite]]
    return parents, elites

def generate_children(parents):
    new_population = []

    first_half_of_parents = parents[:len(parents)//2]
    second_half_of_parents = parents[len(parents)//2:]

    for i in range(len(first_half_of_parents)):
        new_population.append(first_half_of_parents[i][:len(items) // 2] + second_half_of_parents[i][len(items) // 2:])
        new_population.append(first_half_of_parents[i][len(items) // 2:] + second_half_of_parents[i][:len(items) // 2])
        
    return new_population

def mutate(population):
    for individual in population:
        mutation_index = random.randint(0, len(individual)-1)
        individual[mutation_index] = not individual[mutation_index]


items, knapsack_max_capacity = get_big()
print(items)

population_size = 100
generations = 1000
n_selection = 20 # How many parents are chosen
n_elite = 20

start_time = time.time()
best_solution = None
best_fitness = 0
population_history = []
best_history = []

# Create population

population = initial_population(len(items), population_size)
for _ in range(generations):
    population_history.append(population)

    # TODO: implement genetic algorithm

    # Choose parents   

    parents, elites = generate_parents_and_elites(population)

    # Creation of next generation
    
    children = generate_children(parents)
    
    # Mutation

    mutate(children)
    
    # Update of current generation

    population = children + elites

    best_individual, best_individual_fitness = population_best(items, knapsack_max_capacity, population)
    if best_individual_fitness > best_fitness:
        best_solution = best_individual
        best_fitness = best_individual_fitness
    best_history.append(best_fitness)

end_time = time.time()
total_time = end_time - start_time
print('Best solution:', list(compress(items['Name'], best_solution)))
print('Best solution value:', best_fitness)
print('Time: ', total_time)

# plot generations
x = []
y = []
top_best = 10
for i, population in enumerate(population_history):
    plotted_individuals = min(len(population), top_best)
    x.extend([i] * plotted_individuals)
    population_fitnesses = [fitness(items, knapsack_max_capacity, individual) for individual in population]
    population_fitnesses.sort(reverse=True)
    y.extend(population_fitnesses[:plotted_individuals])
plt.scatter(x, y, marker='.')
plt.plot(best_history, 'r')
plt.xlabel('Generation')
plt.ylabel('Fitness')
plt.show()
