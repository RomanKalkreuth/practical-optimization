"""
Implementation of the randomized search algorithm
of the third exercise,

Author:         Roman Kalkreuth, roman.kalkreuth@tu-dortmund.de
                https://orcid.org/0000-0003-1449-5131
                https://ls11-www.cs.tu-dortmund.de/staff/kalkreuth
                https://twitter.com/RomanKalkreuth
"""
import numpy as np
import benchmarks as bm

def randomized_search(f, x0, s0, eps):
    """
    Performs the randomized search
    """
    # Calculate the function value
    fx0 = f(x0)
    k = 0

    # Repeat until s0 is below epsilon
    while s0 >= eps:
        # Calculate a new candidate point by chance
        y = x0 + s0 * np.random.normal(0, 1)

        # Calculate the respective function value
        fy = f(y)

        # Check for improvement and change the candidate point
        # as well as the function value
        if fy < fx0:
            x0 = y
            fx0 = fy
        # If we don't obtain an improvement adjust the
        # step size by bisection
        else:
            s0 = s0 / 2

        k = k + 1

    # Return best solution which we have found so far
    return x0, fx0


rep = 100
s0 = 1
eps = 0.01

res = np.zeros(rep)

# Instantiate an object of the benchmark class
benchmarks = bm.Benchmarks()
f = benchmarks.func31

res = list()

# Iterate over the predefined number of repetitions
for i in range(0,rep-1):
    x0 = np.random.uniform(-10,10)
    r = randomized_search(f, x0, s0, eps)
    res.append(r)

print(res)