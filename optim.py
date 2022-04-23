"""
Example for the use of minimize in scipy.optimize for BFGS
"""

import numpy as np
from scipy.optimize import minimize

def function(X):
    """Test function for the optimization"""
    s = 0
    for i, x in enumerate(X):
        s += x**i
    return s

def gradient(X):
    """Gradient calculation for the test function"""
    grad = np.zeros(len(X))
    for i,x in enumerate(X):
        grad[i] = i*x
    return grad

dim = 10

# define range for input
min, max = 0.0, 10.0

# calculate the starting point uniformly distributed by chance
x0 = np.random.randint(min,max,dim)

# Run the optimization
res = minimize(function, x0, method='BFGS', jac=gradient)

# Get the solution and the function value
sol = res['x']
fs = function(sol)

# Print the total evaluations and the solution
print('Total Evaluations: %d' % res['nfev'])
print('Solution: f(%s) = %.5f' % (sol, fs))


