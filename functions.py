"""
Generates evenly spaced or normal distrbuted data and calulates the function
values which have been used in first and second exercise.

The function are platted using matplotlib and plotly
"""

import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go

def func11(x, y):
    """
    Function 1 from the first exercise sheet (bivariate)
    """
    return 2 * x ** 2 - 3 * x * y + 2 * y ** 2 + 6 * y

def func12(x, y):
    """
    Function 2 from the first exercise sheet (bivariate)
    """
    return 4 * x ** 2 + 4 * x * y + 2 * y ** 2 + 24 * x - 4 * y + 5

def func21(x):
    """
     Function 1 from the first exercise sheet (univariate)
    """
    return (x+3)**2

def func22(x):
    """
    Function 2 from the first exercise sheet (univariate)
    """
    return x ** 2 - 5 * np.cos(10 * x)

def func23(x, y):
    """
    Function 3 from the first exercise sheet (bivariate)
    """
    return x**2 + y**2

def func24(x, y):
    """
    Function 4 from the first exercise sheet (bivariate)
    """
    return x * np.sin(x) + 3 * y ** 2


def gen_normal(vars, mu, sigma, n):
    """
    Generate normal distributed date
    """
    D = np.zeros((vars, n))
    for i in range(0, vars):
        D[i,] = np.random.normal(mu, sigma, n)
    return D

def gen_espaced(vars, min, max, n):
    """
    Generate evenly spaced data
    """
    D = np.zeros((vars, n))
    for i in range(0, vars):
        D[i,] = np.linspace(min, max, n)
    return D

def calc_univariate(D, n , fn):
    """
    Cclculate the function value in univariate case
    """
    Y = np.zeros(n, dtype=D.dtype)
    X = D[0,]

    for i,x in enumerate(X):
        if fn == 21:
            Y[i] = func21(x)
        elif fn == 22:
            Y[i] = func22(x)
        else:
            raise ValueError("Unknown function number")
    return X,Y


def calc_bivariate(D, n , fn):
    """
    Calculate the fucntion values in bivariate case
    """
    Z = np.zeros((n,n), dtype=D.dtype)
    X = D[0,]
    Y = D[1,]

    for i,x in enumerate(X):
        for j,y in enumerate(Y):
            if fn == 11:
                Z[i][j] = func11(x, y)
            elif fn == 12:
                Z[i][j] = func12(x, y)
            elif fn == 23:
                Z[i][j] = func23(x, y)
            elif fn == 24:
                Z[i][j] = func24(x, y)
            else:
                raise ValueError("Unknown function number")
    return X,Y,Z


def plot3d_matplot(X, Y, Z):
    """
    Plot the function in 3D using matplotlib
    """
    figure = plt.figure()
    axes = plt.axes(projection="3d")
    axes.scatter3D(X, Y, Z, c=Z, cmap='cividis');
    axes.view_init(40, 30)
    plt.show()

def plot2d_matplot(X, Y):
    """
    Plot the function in 2D using matplotlib
    """
    plt.plot(X,Y)
    plt.show()


def plot3d_plotly(X, Y, Z):
    """
    Plot the function in 3D using plotly surface
    """
    fig = go.Figure(data=[go.Surface(x=X, y=Y, z=Z)])
    fig.show()


# Function number and number of variables
fn = 22
vars = 1

# Mean and standard deviation for normal distributed data
mu = 0
sigma = 100

# Intervals and step size
max = 10
min = -10
step = 0.1

n = 100
type = "espace"

# Generate data either normal distributed or evenly spaced
if type == "normal":
    D = gen_normal(vars, min, max, n)
elif type == "espace":
    D = gen_espaced(vars, min, max, n)
else:
    raise ValueError("Unknown generation type")

# Distinct between uni- and bivariate case
if vars == 1:
    X, Y = calc_univariate(D, n, fn)
    plot2d_matplot(X,Y)
elif vars == 2:
    X, Y, Z = calc_bivariate(D, n, fn)
    plot3d_plotly(X, Y, Z)
else:
    raise ValueError("Invalid number of variables")