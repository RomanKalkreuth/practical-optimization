"""
Generates equally spaced data and calculates the first derivative.
Pyplot is used to plot the functions.

Author:         Roman Kalkreuth, roman.kalkreuth@tu-dortmund.de
                https://orcid.org/0000-0003-1449-5131
                https://ls11-www.cs.tu-dortmund.de/staff/kalkreuth
                https://twitter.com/RomanKalkreuth
"""

import numpy as np
import matplotlib.pyplot as plt

def generate(start, end, step):
    """ Generates equally spaced data """
    X = np.arange(start, end, step)
    return X

def function(X, func):
    """ Calculates function values for the sine and cosinus function """

    # If the selected function is sine
    if func == "sin":
        # Use numpy to calculate the sine of X
        Y = np.sin(X)
    # Or if the selected function is cosine
    elif func == "cos":
        # Use numpy to calculate the cosine of X
        Y = np.cos(X)
    # Otherwise raise a value error
    else:
        raise ValueError("Unknown function")
    return Y


def derive(X, Y):
    """ Calculates the first derivative of X """

    # Determine the dimension of X
    dim = len(X)

    # Init a numpy array with zeros
    D = np.zeros(dim, dtype=X.dtype)

    # Iterate over X and Y
    for i, (vx, vy) in enumerate(zip(X, Y)):
        if i < dim - 1:
            # Determine the difference for x and y at i and i+1
            dx = X[i] - X[i + 1]
            dy = Y[i] - Y[i + 1]
        # Using dx and dy, calculate the derivation at point i
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
