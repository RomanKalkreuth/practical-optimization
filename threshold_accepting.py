"""
Threshold accepting (TA) implementation from the fifth exercise

Author:         Roman Kalkreuth, roman.kalkreuth@tu-dortmund.de
                https://orcid.org/0000-0003-1449-5131
                https://ls11-www.cs.tu-dortmund.de/staff/kalkreuth
                https://twitter.com/RomanKalkreuth
"""
import numpy as np
import benchmarks as bm


def threshold_accepting(x0: np.array, t0: int, g: float, dim: int, iter: int, func):
    """
    Performs the TA algorithm.

    :param x0: Init parameter vector represented as numpy array
    :param t0: Init T parameter
    :param g: Gamma parameter
    :param dim: Dimension of the problem
    :param iter: Number of iterations
    :param func: Optimization function
    :return: Best parameter and corresponding function value
    """
    # Mean is set to zero
    mean = np.zeros(dim)
    # Covariance matrix is set to the identity matrix
    cov = np.identity(dim)

    # Overtake init x and t values
    xk = x0
    tk = t0

    # Determine init function value
    fxk = func(xk)
    
    k = 0
    # Iterate over the number of iterations
    for i in range(iter):
        # Determine zk from multivariate normal distribution
        zk = np.random.multivariate_normal(mean, cov)

        # Calculate new candidate solution and evaluate it afterwards
        yk = xk + zk
        fyk = func(yk)

        # Test for acceptance
        if fyk <= fxk + t0:
            xk = yk
            fxk = fyk
        # Reduce the acceptance threshold
        tk = tk * g
        k = k + 1

    return xk, fxk


# Meta parameters
dim = 4
iter = 20
t0 = 1
g = 0.5
min = -10.0
max = 10.0

x0 = np.random.uniform(min, max, dim)

# Instantiate an object of the benchmark class
benchmarks = bm.Benchmarks()
func = benchmarks.func51

# Perform the search algorithm
threshold_accepting(x0, t0, g, dim, iter, func)
