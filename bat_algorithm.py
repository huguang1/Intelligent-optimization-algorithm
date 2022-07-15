#!/usr/bin/env python3

"""An implementation of Bat Algorithm
"""

import numpy as np
from numpy.random import random as rand


# Parameters setting
# objfun: objective function
# N_pop: population size, typically 10 to 40
# N_gen: number of generation
# A: loudness (constant or decreasing)
# r: pulse rate (constant or decreasing)
# This frequency range determines the scalings
# You should change these values if necessary
# Qmin: frequency minmum
# Qmax: frequency maxmum
# d: number of dimensions
# lower: lower bound
# upper: upper bound
def bat_algorithm(objfun, N_pop=20, N_gen=1000, A=0.5, r=0.5,
                  Qmin=0, Qmax=2, d=10, lower=-2, upper=2):
    N_iter = 0  # Total number of function evaluations
    # Limit bounds
    Lower_bound = lower * np.ones((1, d))
    Upper_bound = upper * np.ones((1, d))

    Q = np.zeros((N_pop, 1))  # Frequency
    v = np.zeros((N_pop, d))  # Velocities
    S = np.zeros((N_pop, d))

    # Initialize the population/soutions
    # Sol = np.random.uniform(Lower_bound, Upper_bound, (N_pop, d))
    # Fitness = objfun(Sol)
    Sol = np.zeros((N_pop, d))
    Fitness = np.zeros((N_pop, 1))
    for i in range(N_pop):
        Sol[i] = np.random.uniform(Lower_bound, Upper_bound, (1, d))
        Fitness[i] = objfun(Sol[i])

    # Find the initial best solution
    fmin = min(Fitness)
    Index = list(Fitness).index(fmin)
    best = Sol[Index]

    # Start the iterations
    for t in range(N_gen):

        # Loop over all bats/solutions
        for i in range(N_pop):
            # Q[i] = Qmin + (Qmin - Qmax) * np.random.rand
            Q[i] = np.random.uniform(Qmin, Qmax)
            v[i] = v[i] + (Sol[i] - best) * Q[i]
            S[i] = Sol[i] + v[i]
            # Apply simple bounds/limits
            Sol[i] = simplebounds(Sol[i], Lower_bound, Upper_bound)
            # Pulse rate
            if rand() > r:
                # The factor 0.001 limits the step sizes of random walks
                S[i] = best + 0.001 * np.random.randn(1, d)

            # Evaluate new solutions
            # print(i)
            Fnew = objfun(S[i])
            # Update if the solution improves, or not too loud
            if (Fnew <= Fitness[i]) and (rand() < A):
                Sol[i] = S[i]
                Fitness[i] = Fnew

            # update the current best solution
            if Fnew <= fmin:
                best = S[i]
                fmin = Fnew

        N_iter = N_iter + N_pop

    print('Number of evaluations: ', N_iter)
    print("Best = ", best, '\n fmin = ', fmin)

    return best


def simplebounds(s, Lower_bound, Upper_bound):
    Index = s > Lower_bound
    s = Index * s + ~Index * Lower_bound
    Index = s < Upper_bound
    s = Index * s + ~Index * Upper_bound

    return s


# u: array-like
def test_function(u):
    a = u ** 2
    return a.sum(axis=0)


if __name__ == '__main__':
    # print(bat_algorithm(test_function))
    bat_algorithm(test_function)



