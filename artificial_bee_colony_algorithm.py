import math
import random
import matplotlib.pyplot as plt

from typing import List


class ProblemModel:

    def __init__(self, bounds=None):
        self.bounds = bounds

    def getIndependentVar(self):
        if self.bounds is not None:
            independentVar = []
            for bound in self.bounds:
                independentVar.append(bound[0] + random.random() * (bound[1] - bound[0]))
            return independentVar
        else:
            pass

    def getNewVar(self, var_1, var_2):
        if self.bounds is not None:
            newVar = []
            step_random = random.random()
            for v_1, v_2 in zip(var_1, var_2):
                newVar.append(v_1 + step_random * (v_2 - v_1))
            return newVar
        else:
            pass

    def getValue(self, variable):
        if len(variable) == 2:
            x = variable[0]
            y = variable[1]
            return 1 + (x ** 2 + y ** 2) / 4000 - (math.cos(x) * math.cos(y / math.sqrt(2)))
            # return -(x**2-10*math.cos(2*math.pi*x)+10)+(y**2-10*math.cos(2*math.pi*y)+10)
            # return -20*math.exp(-0.2*math.sqrt((x**2+y**2)/2))-math.exp((math.cos(2*math.pi*x)+math.cos(2*math.pi*y))/2)+20+math.e
        else:
            return 1


class NectarSource:
    problem_src = None  # static variable

    def __init__(self, position):
        self.position = position
        self.value = self.problem_src.getValue(position)
        if self.value >= 0:
            self.fitness = 1 / (1 + self.value)
        else:
            self.fitness = 1 + math.fabs(self.value)
        # this definition of fitness means looking for the minimum
        self.trail = 0


class ABCAlgor:
    LIMIT = 10  # If the num of times a source not be updated is more than this, then give up.

    def __init__(self, problem, employedNum, onlookerNum, maxIteration):
        NectarSource.problem_src = problem
        self.problem = problem  # type:ProblemModel
        self.employedNum = employedNum
        self.onlookerNum = onlookerNum
        self.maxIteration = maxIteration
        self.nectarSrc = []  # type:List[NectarSource]
        self.bestNectar = NectarSource(self.problem.getIndependentVar())
        self.resultRecord = []
        for i in range(self.employedNum):
            self.nectarSrc.append(NectarSource(self.problem.getIndependentVar()))

    def updateNectarSrc(self, index):
        # produce a new nectar; if new.fit>old.fit: replace the old; else: old.trail++;
        src = self.nectarSrc[index]
        src_another = random.choice(self.nectarSrc)  # type:NectarSource
        while src_another is src:
            src_another = random.choice(self.nectarSrc)
        src_new = NectarSource(self.problem.getNewVar(src.position, src_another.position))
        if src_new.fitness > src.fitness:
            self.nectarSrc[index] = src_new
        else:
            self.nectarSrc[index].trail += 1

    def employedProcedure(self):
        length = len(self.nectarSrc)  # len(self.nectarSrc) may be changed in self.updateNectarSrc
        for i in range(length):
            self.updateNectarSrc(i)

    def onlookerProcedure(self):
        sum_fitness = 0
        for src in self.nectarSrc:
            sum_fitness += src.fitness
        length = len(self.nectarSrc)
        for i in range(length):
            probability_fit = self.nectarSrc[i].fitness / sum_fitness
            for onlookerBee in range(self.onlookerNum):
                if random.random() < probability_fit:
                    self.updateNectarSrc(i)

    def updateBestNectar(self):
        # use the fitness to select the best, if the problem is finding the max, change the definition of fitness
        for src in self.nectarSrc:
            if src.fitness > self.bestNectar.fitness:
                self.bestNectar = src

    def scoutProcedure(self):
        length = len(self.nectarSrc)
        for i in range(length):
            if self.nectarSrc[i].trail >= self.LIMIT:
                self.nectarSrc[i] = NectarSource(self.problem.getIndependentVar())

    def solve(self):
        for i in range(self.maxIteration):
            self.employedProcedure()
            self.onlookerProcedure()
            self.updateBestNectar()
            self.scoutProcedure()
            self.updateBestNectar()
            self.resultRecord.append(self.bestNectar.value)

    def showResult(self):
        for result in self.resultRecord:
            print(result)
        print('best solution:', self.bestNectar.position)
        print('best value:', self.bestNectar.value)
        plt.plot(self.resultRecord)
        plt.title('result curve')
        plt.tight_layout()
        plt.show()


if __name__ == '__main__':
    beesNum = 100
    employedNum = int(beesNum / 2)
    onlookerNum = int(beesNum / 2)
    maxIteration = 200
    problem = ProblemModel(bounds=[[-10, 10], [-10, 10]])
    abcSolution = ABCAlgor(problem, employedNum, onlookerNum, maxIteration)
    abcSolution.solve()
    abcSolution.showResult()




