class Evaluator:
    def __init__(self, items, max_weight):
        self.items = items
        self.max_weight = max_weight

    def evaluate(self, chromosome):
        total_weight = self.calculateTotalWeight(chromosome)
        total_utility = sum([item[1] * chromosome[i]
                            for i, item in enumerate(self.items)])
        if total_weight > self.max_weight:
            total_utility = 0

        return total_utility

    def calculateTotalWeight(self, chromosome):
        return sum([item[0] * chromosome[i] for i, item in enumerate(self.items)])
