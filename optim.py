"""
Example for the use of minimize in scipy.optimize for BFGS
"""

import numpy as np
from scipy.optimize import minimize

def quadratic_sum(X):
    """ Quadratic sum problem """
    s = 0
    for i, x in enumerate(X):
        s += x ** 2
    return s


def quadratic_sum_scaled(X):
    """ Scaled quadratic sum problem """
    s = 0
    for i, x in enumerate(X):
        s += (i + 1) ** 2 * x ** 2
    return s


def schwefel(X):
    """ Schwefel's problem """
    s1 = 0
    s2 = 0
    n = len(X)
    for i, x in enumerate(X):
        s2 = 0
        for j in range(i, n):
            s2 += x ** 2
    s1 += s2 ** 2
    return s1

dim = 5

# define range for input
min, max = -10.0, 10.0

# calculate the starting point uniformly distributed by chance
xz = np.random.randint(min, max, dim)
res = minimize(schwefel, xz, method='BFGS')

# Get the solution and the function value
sol = res['x']
fs = quadratic_sum(sol)

# Print the total evaluations and the solution
print('Total Evaluations: %d' % res['nfev'])
print('Solution: f(%s) = %.5f' % (sol, fs))



