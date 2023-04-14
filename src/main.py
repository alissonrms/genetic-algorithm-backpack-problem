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
STOP_TIME = 60 * 30
POPULATION_SIZE = 200
INITIAL_MUTATION_RATE = 0.002

EVALUATOR = Evaluator(items, BACKPACK_MAX_WEIGHT)
SELECTION_STRATEGY = selectionTypes['ROULETTE'](EVALUATOR, POPULATION_SIZE)
CROSSOVER_STRATEGY = crossoverTypes['TWO_POINT'](NUM_ITEMS)
MUTATION_STRATEGY = mutationTypes['RANDOM_BIT_BIT'](
    INITIAL_MUTATION_RATE, NUM_ITEMS, EVALUATOR)
MAINTENANCE_STRATEGY = maintenanceTypes['ELITISM']
STOP_STRATEGY = stopTypes['TIME'](MAX_GENERATIONS, STOP_TIME)
POPULATION = Population(POPULATION_SIZE, NUM_ITEMS,
                        EVALUATOR, MAINTENANCE_STRATEGY)


def select_parents_parallel():
    pool = mp.Pool(mp.cpu_count())
    parents = pool.map(SELECTION_STRATEGY.selection, [
                       POPULATION.individuals] * (POPULATION.population_size // 2))
    pool.close()
    pool.join()
    return parents


def crossParents(parents):
    children = []
    for parent1, parent2 in parents:
        child1, child2 = CROSSOVER_STRATEGY.crossover(parent1, parent2)
        children.append(child1)
        children.append(child2)

    return children


programStartTime = time.time()
fitness_history = []
best_solution = max(POPULATION.individuals, key=EVALUATOR.evaluate)

bestSolutionCounter = 0
STOP_STRATEGY.reset()
if __name__ == '__main__':
    while (STOP_STRATEGY.isToContinue()):
        parents = select_parents_parallel()
        children = crossParents(parents)
        if(bestSolutionCounter == 10):
            POPULATION.individuals = MUTATION_STRATEGY.mutate(POPULATION.individuals, EVALUATOR.evaluate(best_solution)) 
            bestSolutionCounter = 0 
        else:
            children = MUTATION_STRATEGY.mutate(children, EVALUATOR.evaluate(best_solution))  
            bestSolutionCounter += 1
            
        POPULATION.adjustPopulation(children)
        POPULATION_SIZE = POPULATION.checkToIncreaseRandomIndividuals(EVALUATOR.evaluate(best_solution))
        SELECTION_STRATEGY.population_size = POPULATION_SIZE

        current_best_solution = max(POPULATION.individuals, key=EVALUATOR.evaluate)
        fitness_history.append(EVALUATOR.evaluate(current_best_solution))
        
        if(EVALUATOR.evaluate(current_best_solution) != EVALUATOR.evaluate(best_solution)):
            best_solution = current_best_solution
            bestSolutionCounter = 0
            
        # print(f"Progresso {STOP_STRATEGY.getProgressPercentage()}%")
        # print(f"Melhor Solução atual: {EVALUATOR.evaluate(best_solution)}")

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

    programEndTime = time.time()
    print(
        f"Tempo execução total: {round((programEndTime - programStartTime))}s")
