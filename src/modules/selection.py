import random
import time
import logging


class SelectionStrategy:
    def __init__(self, evaluator, population_size):
        self.evaluator = evaluator
        self.population_size = population_size
        self.fitnessArray = []
        self.probabilitiesArray = []
        self.total_fitness = 0
    
    def updateFitnessArray(self, population):
        fitness_scores = []
        for chromosome in population:
            fitness_scores.append(self.evaluator.evaluate(chromosome))
            
        self.fitnessArray = fitness_scores
    
    def updateTotalFitness(self):
        self.total_fitness = sum(self.fitnessArray)
        
    def updateProbabilitiesArray(self):
        self.probabilitiesArray = []
        for score in self.fitnessArray:
            self.probabilitiesArray.append(score / self.total_fitness)
    
    def selection(self):
        pass


class RouletteSelectionStrategy(SelectionStrategy):
    def selection(self, population):
        selected_indices = random.choices(
            range(self.population_size), weights=self.probabilitiesArray, k=2)
        
        individuals = []
        for index in selected_indices:
            individuals.append(population[index])

        return individuals


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
