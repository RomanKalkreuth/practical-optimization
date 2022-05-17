"""
Python implementation of the one-lambda-EA from the fifth exercise.

Author:         Roman Kalkreuth, roman.kalkreuth@tu-dortmund.de
                https://orcid.org/0000-0003-1449-5131
                https://ls11-www.cs.tu-dortmund.de/staff/kalkreuth
                https://twitter.com/RomanKalkreuth
"""

from operator import itemgetter

import numpy as np

import benchmarks as bm


def init(dim: int, min: int, max: int) -> np.array:
    """
    Initializes the parent X0 vector with random values.

    :return: Random X0 vector represented as numpy array
    """
    # Numpy array to represent the init vector
    p = np.zeros(dim)

    # Iterate over the dimension pf the problem
    for j in range(dim):
        # Init the vector with uniformly distributed values
        # in the interval [min, max]
        x = np.random.uniform(min, max)
        p[j] = x
    return p


def selection(population: list) -> tuple:
    """
    Selects the best individual from current population.
    Elitist selection strategy.

    :return: Best individual represented as tuple
    """

    # Create a sorted list from our population. Sort the
    # individuals by the respective fitness values
    sorted_population = list(sorted(population, key=itemgetter(2)))

    # The best individual is on top of the sorted list
    best_individual = sorted_population[0]

    return best_individual


def breeding(lam: int, tau: int, parent: tuple, func) -> list:
    """
    Breeds lambda offspring with normal distributed mutation and
    Schwefel's method for step size mutation.

    :param lam: Number of offspring
    :param tau: Tau parameter for stepsize adaption
    :param parent: Parent of the current generation
    :param func: Function of the optimization problem
    :return:
    """
    population = list()

    # Get the parameter vector and the step size of the parent
    p = parent[0]
    s = parent[1]

    i = 0

    # Iterate over lambda
    while i < lam:
        vec = np.zeros(dim)

        # Mutate the step size
        so = s * np.exp(np.random.normal(0, tau ** 2))

        # Iterate over the vector of the parent
        for j, x in enumerate(p):
            # Mutate the each element with the mutated step size
            xo = x + so * np.random.normal()

            # Clip the values within the interval [min, max]
            np.clip(xo, min, max)
            vec[j] = xo
        # Evaluate the new candidate solution
        fitness = evaluation(vec, func)
        # Create a new offspring that is represented with a tuple
        offspring = (vec, so, fitness)

        # Append the new offspring to the population
        population.append(offspring)
        i += 1

    return population


def evaluation(vec: np.array, func) -> float:
    """
    Evaluates a candidate solution against the optimization function.

    :param vec: Parameter vector of the candidate solution
    :param func: Function of the optimization problem
    :return: Fitness value of the candidate solution
    """

    # Get the two parameters for the given optimization function
    x = vec[0]
    y = vec[1]

    # The value of the function is used as the fitness value
    fitness = func(x, y)

    return fitness


def search(dim: int, func, strategy: str, sigma: int = 1.0, min: int = -10, max: int = 10, lam: int = 1,
           evals: int = 200) -> int:
    """
    Performs the 1+lambda evolutionary search algorithm.

    :param dim: Dimension of the problem
    :param func: Function of the optimization problem
    :param strategy: Plus or comma strategy choice
    :param sigma: Standard deviation
    :param min: Minimum constraint
    :param max: Maximum constraint
    :param lam: Numver of offspring (lambda)
    :param evals: Number of fitness evaluation
    :return: Best function parameter obtained after given number of evaluations
    """

    assert dim > 0, "n must be greater than zero"
    assert strategy == "plus" or strategy == "comma", "strategy must be plus or comma"

    # Calculate the tau parameter
    tau = 1 / dim ** 0.5

    # Determine the number of the generations (iterations)
    generations = int(evals / lam)

    # Init the start vector
    x0 = init(dim, min, max)

    # Determine the fitness of the parent
    fitness = func(x0[0], x0[1])

    # The individuals are represented with a tuple
    parent = (x0, sigma, fitness)

    # Iterate over the number of generations
    for i in range(generations):
        population = breeding(lam, tau, parent, func)

        # Add the parent to the selection pool if plus strategy
        # has been selected
        if strategy == "plus":
            population.append(parent)

        # The best individual becomes the parent of the next generation
        best_individual = selection(population)
        parent = best_individual

        print("Generation: " + str(i + 1) + " - Best fitness: " + str(best_individual[2]))

    best_solution = parent[0]

    # Return the best parameter values found so far
    return best_solution[0], best_solution[1]


# Meta parameter for the evolutionary search
min = -10
max = 10
dim = 2
lam = 4
evals = 400
sigma = 1.0
strategy = "plus"

# Instantiate an object of the benchmark class
benchmarks = bm.Benchmarks()
func = benchmarks.func41

# Perform the search algorithm
search(dim, func, strategy, sigma, min, max, lam, evals)
