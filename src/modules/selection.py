import random


class SelectionStrategy:
    def __init__(self, evaluator, population_size):
        self.evaluator = evaluator
        self.population_size = population_size
    
    def selection(self):
        pass


class RouletteSelectionStrategy(SelectionStrategy):
    def selection(self, population):
        fitness_scores = [self.evaluator.evaluate(
            chromosome) for chromosome in population]
        total_fitness = sum(fitness_scores)
        probabilities = [score / total_fitness for score in fitness_scores]
        selected_indices = random.choices(
            range(self.population_size), weights=probabilities, k=2)
        return [population[index] for index in selected_indices]


class RankingSelectionStrategy(SelectionStrategy):
    def selection(self, population):
        fitness_scores = [self.evaluator.evaluate(
            chromosome) for chromosome in population]
        ranked_indices = list(range(len(fitness_scores)))
        ranked_indices.sort(key=lambda x: fitness_scores[x])
        ranked_indices.reverse()
        ranked_probs = [i / (len(ranked_indices) * (len(ranked_indices) + 1) / 2)
                        for i in range(1, len(ranked_indices) + 1)]
        selected_indices = random.choices(
            ranked_indices, weights=ranked_probs, k=2)
        return [population[index] for index in selected_indices]


selectionTypes = {
    'ROULETTE': RouletteSelectionStrategy,
    'RANKING': RankingSelectionStrategy
}
