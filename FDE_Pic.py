import numpy as np
from numpy import *
import random
import math

class Population:
    def __init__(self, min_range, max_range, dim, eta, g_rounds, p_size, object_func, co=0.75, constraint_ueq=tuple()):
        self.min_range = min_range
        self.max_range = max_range
        self.dimension = dim
        self.factor = eta
        self.rounds = g_rounds
        self.size = p_size
        self.cur_round = 1
        self.CO = co
        self.get_object_function_value = object_func

        self.individuality = [np.array([random.uniform(self.min_range[s], self.max_range[s]) for s in range(self.dimension)])
                              for tmp in range(p_size)]
        self.object_function_values = [self.get_object_function_value(v) for v in self.individuality]
        self.mutant = None
        self.phi = random.randint(0, 4)

        self.has_constraint = len(constraint_ueq) > 0
        self.constraint_ueq = list(constraint_ueq)  # a list of unequal constraint functions with ueq[i] <= 0

        self.Best_individuality = []

    def mutate(self):
        self.mutant = []

        for i in range(self.size):
            r0, r1, r2 = 0, 0, 0
            while r0 == r1 or r1 == r2 or r0 == r2 or r0 == i:
                r0 = random.randint(0, self.size - 1)
                r1 = random.randint(0, self.size - 1)
                r2 = random.randint(0, self.size - 1)
            r = (-1)**(random.randint(0, 1))*(i%5)
            theta = i%20
            y = self.individuality[r0] / (2**theta)
            f_i = (-1)**i * (y + y * np.random.randn(self.dimension))

            tar = self.individuality[r0] + multiply(self.individuality[r0], np.random.randn(self.dimension)) * (self.factor ** r)
            tar = tar + f_i

            for j in range(self.dimension):
                if tar[j] > self.max_range[j] or tar[j] < self.min_range[j]:
                    tar[j] = self.individuality[r0][j]
            self.mutant.append(tar)

    def crossover_and_select(self):
        for i in range(self.size):
            Jrand = random.randint(0, self.dimension)
            for j in range(self.dimension):
                if random.random() > (self.CO * math.cos(math.pi * self.phi * self.cur_round / 2)) and j != Jrand:
                    self.mutant[i][j] = self.individuality[i][j]
                tar = self.get_object_function_value(self.mutant[i])
                # unequal constraint
                if self.has_constraint:
                    penalty_ueq = np.array(
                        [np.sum(np.abs([max(0, c_i(x)) for c_i in self.constraint_ueq])) for x in self.individuality])
                    tar = tar + 1e5 * penalty_ueq
                if tar <= self.object_function_values[i]:
                    self.individuality[i] = self.mutant[i]
                    self.object_function_values[i] = tar

    def calculate_best(self):
        m = min(self.object_function_values)
        i = self.object_function_values.index(m)
        self.Best_individuality = self.individuality[i]
        # print("Round：" + str(self.cur_round))
        # print("Best individuality：" + str(self.individuality[i]))
        # print("ObjFuncValue：" + str(m))

    def evolution(self):
        while self.cur_round < self.rounds:
            self.mutate()
            self.crossover_and_select()
            self.cur_round = self.cur_round + 1
        self.calculate_best()
        return self.Best_individuality

# if __name__ == "__main__":
#     def f(v):
#         return -(v[1] + 47) * np.sin(np.sqrt(np.abs(v[1] + (v[0] / 2) + 47))) - v[0] * np.sin(
#             np.sqrt(np.abs(v[0] - v[1] - 47))) - v[2] - v[3] + v[4]
#
#     p = Population(min_range=[-513,-513,-1,-1,0], max_range=[513,513,1,1,3], dim=5, eta=0.8, g_rounds=100, p_size=50,
#           object_func=f, co=0.75, constraint_ueq)
#     print(p.evolution())
