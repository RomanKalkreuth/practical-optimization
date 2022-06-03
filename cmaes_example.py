"""
Example of the CMA-ES mean and covariance matrx update from the sixth exercise sheet.

Author:         Roman Kalkreuth, roman.kalkreuth@tu-dortmund.de
                https://orcid.org/0000-0003-1449-5131
                https://ls11-www.cs.tu-dortmund.de/staff/kalkreuth
                https://twitter.com/RomanKalkreuth
"""

import matplotlib.pyplot as plt
import numpy as np
import benchmarks as bm
import operator as op
from dataclasses import dataclass

@dataclass
class Individual:
    """
    Dataclass which is used to represent an individual for the CMA-ES.

    x and y are the function parameters.

    z is the value of the objective function which is
    represents the fitness.
    """
    x: float
    y: float
    z: float


def update_cov(parents: list, Ct: np.array, mt: np.array, w: int, mu: int, eta: int) -> np.array:
    """
    Update of the covariance matrix.

    :param parents: List of parents
    :param Ct: Current covariance matrix
    :param mt: Current mean
    :param w: Weight
    :param mu: Number of parents
    :param eta: Learning rate
    :return: Updated covariance matrix (Ct1)
    """

    s = 0
    # Iterate over the parent individuals
    for i, ind in enumerate(parents):
        # Calculate the distance to the mean in each dimension
        dx = ind.x - mt[0]
        dy = ind.y - mt[1]
        # Create distance vector
        dxy = np.array([dx, dy])
        # Tranpose the vector
        tdxy = dxy.transpose()
        # Summarize the product the weight and distance vectors
        s += w * dxy * tdxy
    # Finally, update the covariance matrix
    Ct1 = (1 - eta) * Ct + eta * s
    return Ct1


def update_mean(parents: list, mu: int) -> np.array:
    """
    Updates the mean using the mu-best parents.

    :param parents: List of parents
    :param mu: Number of parents

    :return:updated mean
    """
    # Variables for calculating the mean in the given dimensions
    sx = 0
    sy = 0

    # Sum the x and y values of all parents
    for ind in parents:
        sx += ind.x
        sy += ind.y

    # Division by the number of parents
    sx /= mu
    sy /= mu

    # Create the new mean array
    mean = np.array([sx, sy])

    return mean


def selection(population: list, mu: int) -> (list, list):
    """
    Selects the mu-best individuals from the populations.

    :param population: List of individuals
    :param mu: Number of parents

    :return: Sublist with mu-best individuals
    """

    # Create a sorted list from our population. Sort the
    # individuals by the respective fitness values
    sorted_population = list(sorted(population, key=op.attrgetter("z")))

    # Get number of individuals
    n = len(sorted_population)

    # Return a sublists that contain the mu-best parents and the rest
    parents = sorted_population[0:mu]
    other = sorted_population[mu:n]
    return parents, other


def evaluate(d: np.array, func) -> int:
    """
    Fitness evaluation function that calculates the value
    of the given objective function.

    :param d: Function parameters
    :param func: Objective function
    :return: Function value
    """
    return func(d)


def breed(lmb: int, func, Ct: np.array, mt: np.array) -> list:
    """
    Breeds lambda offspring by mutating the mean with multivariate normal distribution.

    :param lmb: Number of offspring
    :param func: Objective function
    :param Ct: Covariance matrix
    :param mt: Mean of the parents
    :return: List of offspring
    """
    offspring = []

    # Iterate over the number of parents
    for i in range(lmb):
        # Breed a new candidate using mt and Ct
        d = np.random.multivariate_normal(mt, Ct)
        # Evaluate the candidate solution
        z = evaluate(d, func)
        # Create the offspring
        ind = Individual(d[0], d[1], z)
        # Append the offspring to the list
        offspring.append(ind)
    return offspring


def plot(parents: list, other: list, mt: np.array, mt1: np.array, Ct: np.array, Ct1: np.array):
    """
    Plot the offspring, selected parents, means and ellipses.

    :param parents: List of parents
    :param other: List of remaining individuals (excluding parents)
    :param mt: Initial mean
    :param mt1: Updated mean
    :param Ct: Initial covariance matrix (identity matrix)
    :param Ct1: Updated Covariance matrix
    """

    # Get dimensions
    n = len(parents)
    no = len(other)

    # Init arrays for the coordinates and function values of the parents
    px = np.zeros(n)
    py = np.zeros(n)
    pz = np.zeros(n)

    # ... and individuals
    ox = np.zeros(no)
    oy = np.zeros(no)
    oz = np.zeros(no)


    # Fill the arrays with the respective values

    for i, ind in enumerate(parents):
        px[i] = ind.x
        py[i] = ind.y
        pz[i] = ind.z

    for i, ind in enumerate(other):
        ox[i] = ind.x
        oy[i] = ind.y
        oz[i] = ind.z

    # Create scatter plots for the visualization of the individuals
    plt.scatter(px, py, c="red", label="parents")
    plt.scatter(ox, oy, c="blue", label="other")
    plt.scatter(mt[0], mt[1], s=100, c="black", marker="P", label="mt")
    plt.scatter(mt1[0], mt1[1], s=100, c="green", marker="P", label="mt1")

    # Calculate the ellipses for Ct and Ct1
    e1x, e1y = ellipse(Ct)
    e2x, e2y = ellipse(Ct1)

    # Plot the ellipses
    plt.plot(e1x + mt[0], e1y + mt[1], c="black")
    plt.plot(e2x + mt1[0], e2y + mt1[1], c="green")

    plt.legend()
    plt.show()


def ellipse(C: np.array):
    """
    Draws the corresponding 95% prediction ellipsis of C by using its eigenvalues and eigenvectors.

    Random points drawn from multivariate normal distribution fall inside this ellipse with
    a probability of 95%

    Mahalanobis distance and chi-squared distribution are used to scale up the ellipse.

    :param C: Covariance matrix
    :return: Corresponding ellipsis
    """
    # Get the eigenvalues and eigenvectors of C
    eigenvalues, eigenvectors = np.linalg.eig(C)

    # Calculate 1000 evenly spaced points in the inverval [0, 2 * oi]
    theta = np.linspace(0, 2 * np.pi, 1000);

    # Use the cumulative chi-squared distribution to scale up the ellipse for 95% confidence
    CHI95 = 5.991

    # Calculate the square root for each eigenvalue and scale it with the 95% chi-squared distribution value
    omega = np.sqrt(CHI95*eigenvalues)

    # Create arrays for x and y values of the ellipsis
    ex = np.array(theta)
    ey = np.array(theta)

    # Iterate over theta
    for i, t in enumerate(theta):
      # Parametric equation
      gamma = eigenvectors @ (omega * [np.sin(t), np.cos(t)])

      # Store x and y values
      ex[i] = gamma[0]
      ey[i] = gamma[1]

    return ex,ey


# Meta parameters
mu = 10
lmb = 100
w = 1 / mu
eta = 0.2
dim = 2

# Init mean and covariance matrix
mt = np.array([3, 3])
Ct = np.identity(dim)

# Instantiate the benchmark functions
benchmarks = bm.Benchmarks()
func = benchmarks.func62

offspring = breed(lmb, func, Ct, mt)
parents, other = selection(offspring, mu)

Ct1 = update_cov(parents, Ct, mt, w, mu, eta)
mt1 = update_mean(parents, mu)

plot(parents, other, mt, mt1, Ct, Ct1)