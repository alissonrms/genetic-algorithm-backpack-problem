import random
import csv
import matplotlib.pyplot as plt

# Ler a lista de itens
items = []
with open('items.csv', newline='') as csvfile:
    reader = csv.DictReader(csvfile, delimiter=';')
    for row in reader:
        peso = float(row['Peso'].replace(',', '.'))
        utilidade = int(row['Utilidade'])
        item = (peso, utilidade)
        items.append(item)

# Definir os parâmetros do problema
MAX_WEIGHT = 50000
NUM_ITEMS = len(items)
MAX_GENERATIONS = 10000
POPULATION_SIZE = 50
MUTATION_RATE = 0.1

# Criar a população inicial
population = []
for i in range(POPULATION_SIZE):
    chromosome = [random.randint(0, 1) for _ in range(NUM_ITEMS)]
    population.append(chromosome)

# Definir a função de avaliação


def evaluate(chromosome):
    total_weight = sum([item[0] * chromosome[i]
                       for i, item in enumerate(items)])
    total_utility = sum([item[1] * chromosome[i]
                        for i, item in enumerate(items)])
    if total_weight > MAX_WEIGHT:
        total_utility = 0
    return total_utility

# Definir a função de seleção


def selection(population):
    fitness_scores = [evaluate(chromosome) for chromosome in population]
    total_fitness = sum(fitness_scores)
    probabilities = [score / total_fitness for score in fitness_scores]
    selected_indices = random.choices(
        range(POPULATION_SIZE), weights=probabilities, k=2)
    return [population[index] for index in selected_indices]

# Definir a função de cruzamento


def crossover(parent1, parent2):
    crossover_point = random.randint(1, NUM_ITEMS - 1)
    child1 = parent1[:crossover_point] + parent2[crossover_point:]
    child2 = parent2[:crossover_point] + parent1[crossover_point:]
    return child1, child2

# Definir a função de mutação


def mutation(chromosome):
    for i in range(NUM_ITEMS):
        if random.random() < MUTATION_RATE:
            chromosome[i] = 1 - chromosome[i]
    return chromosome


# Lista com os valores de fitness por geração
fitness_history = []

# Rodar o algoritmo genético
for generation in range(MAX_GENERATIONS):
    # Selecionar os pais
    parents = [selection(population) for _ in range(POPULATION_SIZE // 2)]

    # Cruzar os pais para gerar filhos
    children = []
    for parent1, parent2 in parents:
        child1, child2 = crossover(parent1, parent2)
        children.append(mutation(child1))
        children.append(mutation(child2))

    # Avaliar a população atual e a nova geração
    current_population = population
    current_fitness = [evaluate(chromosome)
                       for chromosome in current_population]
    new_population = current_population + children
    new_fitness = [evaluate(chromosome) for chromosome in new_population]

    # Selecionar a próxima geração
    population = []
    for i in range(POPULATION_SIZE):
        max_index = max(range(len(new_population)),
                        key=lambda index: new_fitness[index])
        population.append(new_population[max_index])
        del new_population[max_index]
        del new_fitness[max_index]

    # Imprimir a melhor solução encontrada até agora
    best_solution = max(population, key=evaluate)
    fitness_history.append(evaluate(best_solution))

# Plotar o gráfico de linha
plt.plot(fitness_history)
plt.title('Fitness por Geração')
plt.xlabel('Geração')
plt.ylabel('Fitness')
plt.show()

best_solution = max(population, key=evaluate)
print(f"Fitness = {evaluate(best_solution)}")
