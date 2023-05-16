import random
import time
import matplotlib.pyplot as plt
import multiprocessing as mp

from modules.evaluator import Evaluator
from modules.selection import selectionTypes
from modules.crossover import crossoverTypes
from modules.mutation import mutationTypes
from modules.stop import stopTypes
from modules.dataInput import readData
from modules.population import Population, maintenanceTypes

items = readData('items.csv')

BACKPACK_MAX_WEIGHT = 30
NUM_ITEMS = len(items)
MAX_GENERATIONS = 10
STOP_TIME = 60 * 20
POPULATION_SIZE = 200
INITIAL_MUTATION_RATE = 0.02

EVALUATOR = Evaluator(items, BACKPACK_MAX_WEIGHT)
SELECTION_STRATEGY = selectionTypes['ROULETTE'](EVALUATOR, POPULATION_SIZE)
CROSSOVER_STRATEGY = crossoverTypes['TWO_POINT'](NUM_ITEMS)
MUTATION_STRATEGY = mutationTypes['RANDOM_BIT_BIT'](
    INITIAL_MUTATION_RATE, NUM_ITEMS, EVALUATOR)
MAINTENANCE_STRATEGY = maintenanceTypes['REPLACEMENT']
STOP_STRATEGY = stopTypes['TIME'](MAX_GENERATIONS, STOP_TIME)
POPULATION = Population(POPULATION_SIZE, NUM_ITEMS,
                        EVALUATOR, MAINTENANCE_STRATEGY)

K = 10
Y = 20
M = 2


def select_parents_parallel():
    pool = mp.Pool(mp.cpu_count())
    parents = pool.map(SELECTION_STRATEGY.selection, [
                       POPULATION.individuals] * (POPULATION.population_size // 2))

    parentsToReturn = []
    for parent in parents:
        parentsToReturn.append(parent[0])
        parentsToReturn.append(parent[1])

    pool.close()
    pool.join()
    return parentsToReturn


def crossParents(parents):
    children = []
    for i in range(0, len(parents), 2):
        child1, child2 = CROSSOVER_STRATEGY.crossover(
            parents[i], parents[i + 1])
        children.append(child1)
        children.append(child2)
    return children


def verificar_convergencia():
    numero_conjuntos = 0
    individuos_grupos = []  # Lista de indivíduos que representam os grupos
    quantidade_individuos_grupos = []  # Lista de quantidade de indivíduos que cada grupo possui

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
        print(quantidade_individuos_grupos)
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

bestSolutionCounter = 0
increasePopulationNextTime = False

STOP_STRATEGY.reset()
if __name__ == '__main__':
    while (STOP_STRATEGY.isToContinue()):
        parents = select_parents_parallel()
        children = crossParents(parents)
        children = MUTATION_STRATEGY.mutate(
            children, EVALUATOR.evaluate(best_solution))

        POPULATION.adjustPopulation(children)
        SELECTION_STRATEGY.population_size = POPULATION_SIZE

        current_best_solution = max(
            POPULATION.individuals, key=EVALUATOR.evaluate)
        fitness_history.append(EVALUATOR.evaluate(current_best_solution))

        if (EVALUATOR.evaluate(current_best_solution) != EVALUATOR.evaluate(best_solution)):
            best_solution = current_best_solution
            bestSolutionCounter = 0

        print(f"Progresso {STOP_STRATEGY.getProgressPercentage()}%")
        print(f"Melhor Solução atual: {EVALUATOR.evaluate(best_solution)}")
        if (len(fitness_history) % 10 == 0):
            if (verificar_convergencia()):
                if(increasePopulationNextTime):
                    POPULATION_SIZE = POPULATION.checkToIncreaseRandomIndividuals()
                    SELECTION_STRATEGY.population_size = POPULATION_SIZE
                    increasePopulationNextTime = False
                else:
                    POPULATION.individuals = MUTATION_STRATEGY.mutate(
                        POPULATION.individuals, EVALUATOR.evaluate(best_solution))
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
