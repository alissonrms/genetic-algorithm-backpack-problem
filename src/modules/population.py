import random


class Population:
    def __init__(self, population_size, num_items, evaluator, maintenance_strategy):
        self.population_size = population_size
        self.num_items = num_items
        self.evaluator = evaluator
        self.maintenance_strategy = maintenance_strategy(
            population_size, evaluator)
        self.individuals = []
        self.generations = 1
        self.bestSolution = 0
        self.sameSolutionCounter = 0
        self.__createPopulation()

    def __createPopulation(self):
        self.individuals = self.generateRandomIdividuals(self.population_size)

    def generateRandomIdividuals(self, size):
        population = []
        for _ in range(size):
            chromosome = random.choices(
                [0, 1], weights=[0.96, 0.04], k=self.num_items)
            population.append(chromosome)
        return population

    def adjustPopulation(self, children):
        self.individuals = self.maintenance_strategy.execute(
            self.individuals, children)
        self.generations += 1

    def checkToIncreaseRandomIndividuals(self):
        if self.population_size < 1000:
                print('Aumentando população')
                self.population_size += int(self.population_size * 0.5)
                self.maintenance_strategy.population_size = self.population_size
                self.individuals = self.maintenance_strategy.execute(
                    self.individuals, self.generateRandomIdividuals(int(self.population_size * 0.5)))

        return self.population_size


class MaintenanceStrategy:
    def __init__(self, population_size, evaluator):
        self.population_size = population_size
        self.evaluator = evaluator

    def execute(self):
        pass


class ReplacementMaintenanceStrategy(MaintenanceStrategy):
    def __init__(self, population_size, evaluator):
        super().__init__(population_size, evaluator)

    def execute(self, population, new_individuals):
        current_population = population
        new_population = current_population + new_individuals
        new_fitness = [self.evaluator.evaluate(chromosome)
                       for chromosome in new_population]

        population = []
        for _ in range(self.population_size):
            max_index = max(range(len(new_population)),
                            key=lambda index: new_fitness[index])
            population.append(new_population[max_index])
            del new_population[max_index]
            del new_fitness[max_index]

        return population


class ElitismMaintenanceStrategy(MaintenanceStrategy):
    def __init__(self, population_size, evaluator):
        super().__init__(population_size, evaluator)

    def execute(self, population, new_individuals):
        elite_size = int(0.01 * self.population_size)
        current_population = population
        new_population = current_population + new_individuals
        new_fitness = [self.evaluator.evaluate(
            chromosome) for chromosome in new_population]

        elite_population = []
        elite_fitness = []

        for _ in range(elite_size):
            max_index = max(range(len(current_population)), key=lambda index: self.evaluator.evaluate(
                current_population[index]))
            elite_population.append(current_population[max_index])
            elite_fitness.append(self.evaluator.evaluate(
                current_population[max_index]))
            del current_population[max_index]

        population = elite_population

        for i in range(self.population_size - elite_size):
            max_index = max(range(len(new_population)),
                            key=lambda index: new_fitness[index])
            population.append(new_population[max_index])
            del new_population[max_index]
            del new_fitness[max_index]

        return population


maintenanceTypes = {
    'REPLACEMENT': ReplacementMaintenanceStrategy,
    'ELITISM': ElitismMaintenanceStrategy
}
