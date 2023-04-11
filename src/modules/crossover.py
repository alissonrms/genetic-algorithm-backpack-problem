import random


class CrossoverStrategy:
    def __init__(self, num_chromosomes):
        self.num_chromosomes = num_chromosomes

    def crossover(self):
        pass


class OnePointStrategy(CrossoverStrategy):
    def __init__(self, num_chromosomes):
        super().__init__(num_chromosomes)

    def crossover(self, parent1, parent2):
        crossover_point = random.randint(1, self.num_chromosomes - 1)
        child1 = parent1[:crossover_point] + parent2[crossover_point:]
        child2 = parent2[:crossover_point] + parent1[crossover_point:]
        return child1, child2


class TwoPointStrategy(CrossoverStrategy):
    def __init__(self, num_chromosomes):
        super().__init__(num_chromosomes)

    def crossover(self, parent1, parent2):
        crossover_point1 = random.randint(1, self.num_chromosomes - 2)
        crossover_point2 = random.randint(
            crossover_point1, self.num_chromosomes - 1)
        child1 = parent1[:crossover_point1] + \
            parent2[crossover_point1:crossover_point2] + \
            parent1[crossover_point2:]
        child2 = parent2[:crossover_point1] + \
            parent1[crossover_point1:crossover_point2] + \
            parent2[crossover_point2:]
        return child1, child2


crossoverTypes = {
    'ONE_POINT': OnePointStrategy,
    'TWO_POINT': TwoPointStrategy
}
