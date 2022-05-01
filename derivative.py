"""
Generates equally spaced data and calculates the first derivative.
Pyplot is used to plot the functions."
"""

import numpy as np
import matplotlib.pyplot as plt

def generate(start, end, step):
    """ Generates equally spaced data """
    X = np.arange(start, end, step)
    return X

def function(X, func):
    """ Calculates function values for the sine and cosinus function """
    if func == "sin":
        Y = np.sin(X)
    elif func == "cos":
        Y = np.cos(X)
    else:
        raise ValueError("Unknown function")
    return Y


def derive(X, Y):
    """ Calculates the first derivative of X """
    dim = len(X)
    D = np.zeros(dim, dtype=X.dtype)
    for i, (vx, vy) in enumerate(zip(X, Y)):
        if i < (dim - 1):
            dx = X[i] - X[i + 1]
            dy = Y[i] - Y[i + 1]
        D[i] = dy / dx
    return D

# Literals for the generation of the X data
fac = 4
step  = 0.1

# Generate data from 0 two fac * pi equally spaced with
# predefined step size (step)
X = generate(0, fac * np.pi, step)

# Calculate sine and cosine values
Y1 = function(X, "sin")
Y2 = function(X, "cos")

# Derive from the sine function
D = derive(X, Y1)

# Initialise subplot using number of rows and columns
fig, ax = plt.subplots(1, 2)

# Plot the sine function
ax[0].plot(X, Y1)
ax[0].set_title("Sine function")

# For Cosine Function
ax[1].plot(X, D)
ax[1].set_title("First derivative")

# Combine the plots and show
plt.show()
