import random


class MutationStrategy:
    def __init__(self, initialMutationRate, num_chromosomes, evaluator):
        self.initialMutationRate = initialMutationRate
        self.mutationRate = initialMutationRate
        self.num_chromosomes = num_chromosomes
        self.evaluator = evaluator
        self.bestSolutionBuffer = 0
        self.sameSolutionCounter = 0

    def checkToChangeMutationRate(self, current_best_solution):

        if current_best_solution == self.bestSolutionBuffer:
            self.sameSolutionCounter += 1
            if self.sameSolutionCounter >= 10:
                self.mutationRate = min(self.mutationRate + 0.001, 0.1)
                self.sameSolutionCounter = 0
        else:
            self.mutationRate = self.initialMutationRate
            self.sameSolutionCounter = 0
            self.bestSolutionBuffer = current_best_solution
    
    def mutate(self):
        pass
    


class RandomMutationStrategy(MutationStrategy):
    def __init__(self, initialMutationRate, num_chromosomes, evaluator):
        super().__init__(initialMutationRate, num_chromosomes, evaluator)

    def mutate(self, population, current_best_solution):
        for chromosome in population:
            for i in range(len(chromosome)):
                if random.random() < self.mutationRate:
                    if(random.random() > 0.1):
                        chromosome[i] = 0
                    else:
                        chromosome[i] = 1


        self.checkToChangeMutationRate(current_best_solution)
        return population


class SelectiveMutationStrategy(MutationStrategy):
    def __init__(self, initialMutationRate, num_chromosomes, evaluator):
        super().__init__(initialMutationRate, num_chromosomes, evaluator)

    def mutate(self, population, current_best_solution):
        selected_individuals = random.sample(population, 10)

        for individual in selected_individuals:
            for i in range(len(individual)):
                if random.random() < self.mutationRate:
                    if(random.random() > 0.1):
                        individual[i] = 0
                    else:
                        individual[i] = 1

        self.checkToChangeMutationRate(current_best_solution)
        return population


mutationTypes = {
    'RANDOM_BIT_BIT': RandomMutationStrategy,
    'SELECTIVE': SelectiveMutationStrategy
}
