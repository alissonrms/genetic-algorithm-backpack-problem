import time


class StopStrategy:
    def __init__(self, max_generations, stop_time):
        self.max_generations = max_generations
        self.stop_time = stop_time

    def isToContinue(self) -> bool:
        pass

    def reset(self):
        pass

    def getProgressPercentage(self):
        pass


class GenerationsStopStrategy(StopStrategy):
    def __init__(self, max_generations, stop_time):
        self.counter = 0
        super().__init__(max_generations, stop_time)

    def isToContinue(self):
        if (self.counter >= self.max_generations):
            return False

        self.counter += 1
        return True

    def reset(self):
        self.counter = 0

    def getProgressPercentage(self):
        return int(self.counter / self.max_generations * 100)


class TimeStopStrategy(StopStrategy):
    def __init__(self, max_generations, stop_time):
        self.startTime = time.time()
        super().__init__(max_generations, stop_time)

    def isToContinue(self):
        if (time.time() >= self.stop_time + self.startTime):
            return False

        return True

    def reset(self):
        self.startTime = time.time()

    def getProgressPercentage(self):
        return min(int((time.time() - self.startTime) / (self.stop_time) * 100), 100)


stopTypes = {
    'MAX_GENERATIONS': GenerationsStopStrategy,
    'TIME': TimeStopStrategy
}
