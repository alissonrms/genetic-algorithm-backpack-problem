import random
import time
import matplotlib.pyplot as plt
import multiprocessing as mp
import logging

from modules.evaluator import Evaluator
from modules.selection import selectionTypes
from modules.crossover import crossoverTypes
from modules.mutation import mutationTypes
from modules.stop import stopTypes
from modules.dataInput import readData
from modules.population import Population, maintenanceTypes

items = readData('items.csv')
logging.basicConfig(filename='logs.txt', level=logging.DEBUG,
                    format='%(asctime)s %(levelname)s: %(message)s')

BACKPACK_MAX_WEIGHT = 30
NUM_ITEMS = len(items)
MAX_GENERATIONS = 10
STOP_TIME = 60 * 30
POPULATION_SIZE = 200
INITIAL_MUTATION_RATE = 0.005

EVALUATOR = Evaluator(items, BACKPACK_MAX_WEIGHT)
SELECTION_STRATEGY = selectionTypes['ROULETTE'](EVALUATOR, POPULATION_SIZE)
CROSSOVER_STRATEGY = crossoverTypes['ONE_POINT'](NUM_ITEMS)
MUTATION_STRATEGY = mutationTypes['RANDOM_BIT_BIT'](
    INITIAL_MUTATION_RATE, NUM_ITEMS, EVALUATOR)
MAINTENANCE_STRATEGY = maintenanceTypes['REPLACEMENT']
STOP_STRATEGY = stopTypes['TIME'](MAX_GENERATIONS, STOP_TIME)
POPULATION = Population(POPULATION_SIZE, NUM_ITEMS,
                        EVALUATOR, MAINTENANCE_STRATEGY)

K = 5
Y = 3
M = 120


def selectParents():
    selectedParents = []
    for _ in range(POPULATION.population_size // 2):
        selectedIndividuals = SELECTION_STRATEGY.selection(
            POPULATION.individuals)
        selectedParents.append(selectedIndividuals[0])
        selectedParents.append(selectedIndividuals[1])

    return selectedParents


def crossParents(parents):
    children = []
    for i in range(0, len(parents), 2):
        child1, child2 = CROSSOVER_STRATEGY.crossover(
            parents[i], parents[i + 1])
        children.append(child1)
        children.append(child2)
    return children


def applyMutationToAllPopulation():
    pop_size = len(POPULATION.individuals)
    num_preserved = int(0.1 * pop_size)
    
    fitness_values = [EVALUATOR.evaluate(individual) for individual in POPULATION.individuals]
    sorted_population = [x for _, x in sorted(zip(fitness_values, POPULATION.individuals), reverse=True)]
    preserved_individuals = sorted_population[:num_preserved]
    next_generation = preserved_individuals
    
    mutated_individuals = MUTATION_STRATEGY.mutate(sorted_population[num_preserved:], EVALUATOR.evaluate(best_solution))
    next_generation.extend(mutated_individuals)
    logging.info(f"Nova população {next_generation}");
    
    POPULATION.individuals = next_generation    


def verificar_convergencia():
    numero_conjuntos = 0
    individuos_grupos = []  # Lista de indivíduos que representam os grupos
    # Lista de quantidade de indivíduos que cada grupo possui
    quantidade_individuos_grupos = []

    i = 0
    while i < POPULATION.population_size and numero_conjuntos < K:
        j = 1
        while j <= numero_conjuntos:
            if distancia_individuos(POPULATION.individuals[i], individuos_grupos[j - 1]) < Y:
                # Incrementa o número de indivíduos do grupo j
                quantidade_individuos_grupos[j - 1] += 1

                if quantidade_individuos_grupos[j - 1] > M:
                    # Há convergência
                    print(f"Há convergência por muitos individuos no mesmo conjunto")
                    return True

                break

            j += 1

        if j > numero_conjuntos:
            numero_conjuntos += 1
            # Cria um novo grupo com o indivíduo (i)
            individuos_grupos.append(POPULATION.individuals[i])
            quantidade_individuos_grupos.append(1)

        i += 1

    if numero_conjuntos < K:
        # Há convergência
        print(f"Há convergência por poucos conjuntos")
        return True
    else:
        # Não há convergência
        print(f"Não há convergência")
        return False


def distancia_individuos(individuo, grupo):
    distancia = 0
    for gene_individuo, gene_grupo in zip(individuo, grupo):
        if gene_individuo != gene_grupo:
            distancia += 1
    return distancia


programStartTime = time.time()
fitness_history = []
best_solution = max(POPULATION.individuals, key=EVALUATOR.evaluate)

increasePopulationNextTime = False

STOP_STRATEGY.reset()
if __name__ == '__main__':
    while (STOP_STRATEGY.isToContinue() and EVALUATOR.evaluate(best_solution) != 21312):
        start_time = time.time()
        SELECTION_STRATEGY.updateFitnessArray(POPULATION.individuals)
        SELECTION_STRATEGY.updateTotalFitness()
        SELECTION_STRATEGY.updateProbabilitiesArray()
        execution_time = time.time() - start_time
        logging.info(
            f"As funcoes de definicao de fitness demoraram {execution_time} segundos para ser executadas.")

        start_time = time.time()
        parents = selectParents()
        execution_time = time.time() - start_time
        logging.info(
            f"A função 'select_parents_parallel' levou {execution_time} segundos para ser executada.")

        start_time = time.time()
        children = crossParents(parents)
        execution_time = time.time() - start_time
        logging.info(
            f"A função 'crossParents' levou {execution_time} segundos para ser executada.")

        start_time = time.time()
        children = MUTATION_STRATEGY.mutate(
            children, EVALUATOR.evaluate(best_solution))
        execution_time = time.time() - start_time
        logging.info(
            f"A função 'mutate' levou {execution_time} segundos para ser executada.")

        start_time = time.time()
        POPULATION.adjustPopulation(children)
        execution_time = time.time() - start_time
        logging.info(
            f"A função 'adjustPopulation' levou {execution_time} segundos para ser executada.")

        start_time = time.time()
        SELECTION_STRATEGY.population_size = POPULATION_SIZE
        execution_time = time.time() - start_time
        logging.info(
            f"A atribuição de 'population_size' levou {execution_time} segundos para ser executada.")

        start_time = time.time()
        current_best_solution = max(
            POPULATION.individuals, key=EVALUATOR.evaluate)
        execution_time = time.time() - start_time
        logging.info(
            f"A função 'max' levou {execution_time} segundos para ser executada.")

        start_time = time.time()
        fitness_history.append(EVALUATOR.evaluate(current_best_solution))
        execution_time = time.time() - start_time
        logging.info(
            f"A função 'append' levou {execution_time} segundos para ser executada.")

        if (EVALUATOR.evaluate(current_best_solution) > EVALUATOR.evaluate(best_solution)):
            best_solution = current_best_solution

        print(f"Progresso {STOP_STRATEGY.getProgressPercentage()}%")
        print(
            f"Melhor Solução atual: {EVALUATOR.evaluate(current_best_solution)}")
        if (len(fitness_history) % 10 == 0):
            if (verificar_convergencia()):
                if (increasePopulationNextTime):
                    POPULATION_SIZE = POPULATION.checkToIncreaseRandomIndividuals()
                    SELECTION_STRATEGY.population_size = POPULATION_SIZE
                    increasePopulationNextTime = False
                else:
                    applyMutationToAllPopulation()
                    increasePopulationNextTime = True

    # Plotar o gráfico de linha
    plt.plot(fitness_history)
    plt.title('Fitness por Geração')
    plt.xlabel('Geração')
    plt.ylabel('Fitness')
    plt.show()
    print(f"Melhor solução: {EVALUATOR.evaluate(best_solution)} de utilidade")
    print(
        f"Peso melhor solução: {EVALUATOR.calculateTotalWeight(best_solution):.3f}kg")
    print(f"Quantidade de gerações: {len(fitness_history)}")
    print(f"Tamanho da população: {POPULATION.population_size} individuos")

    programEndTime = time.time()
    print(
        f"Tempo execução total: {round((programEndTime - programStartTime))}s")
