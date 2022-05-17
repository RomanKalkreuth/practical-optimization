"""
This Python implementation of the compass search adapts the version
from the third exercise.

Author:         Roman Kalkreuth, roman.kalkreuth@tu-dortmund.de
                https://orcid.org/0000-0003-1449-5131
                https://ls11-www.cs.tu-dortmund.de/staff/kalkreuth
                https://twitter.com/RomanKalkreuth
"""

import numpy as np
import benchmarks as bm


def directions(x0):
    """
    Calculate the directions for x0
    """
    D = np.diag(x0)
    D = np.concatenate((D, np.negative(D)), axis=1)
    return D


def search(f, x0, s0, theta, tol=0.001, opt=0, kmax=1000):
    """
    Performs the compass search in respect to a
    """

    # Get possible directions:
    D = directions(x0)
    k = 0

    # Calculate init vector
    fx = f(x0)

    # Repeat until fx0 has reached a solution which
    # meets our tolerance criterion
    while fx > abs(opt - tol):
        updated = False

        # Iterate over the dimensions
        for col in range(D.shape[1]):
            # Calculate new candidate point and function value
            y = x0 + s0 * col
            fy = f(y)

            # Check for improvement and perform update if the
            # candidate value has been improved
            if fy < fx:
                x0 = y
                fx = fy
                updated = True
                # Break out of loop after update
                break

        # If not updated has been performed, just adjust the
        # step size
        if not updated:
            s0 = theta * s0

        # Increase the iterator counter
        k = k + 1

        print("Iteration: " + str(k) + " - Function value: " + str(fx))

        # Trigger the max evaluation termination criterion when
        # the evaluation budget has been exceeded
        if k >= kmax:
            print("Optimum not found")
            break

    return x0, fx, k


# Define dimension
dim = 10

# Define range for input
xmin, xmax = -10.0, 10.0

# Define meta parameters
xtol = 0.0
dtol = 0.0
dinit = 0.0
kmax = 1000

s0 = 2
theta = 0.5

# Instantiate an object of the benchmark class
benchmarks = bm.Benchmarks()
f = benchmarks.schwefel

# Choose the init parameter values uniformly at random
x0 = np.random.randint(xmin, xmax, dim)

# Perform the search
res = search(f, x0, s0, theta)