"""
Benchmark class that contains the benchmarking functions
for the exercises

Author:         Roman Kalkreuth, roman.kalkreuth@tu-dortmund.de
                https://orcid.org/0000-0003-1449-5131
                https://ls11-www.cs.tu-dortmund.de/staff/kalkreuth
                https://twitter.com/RomanKalkreuth
"""


import math
import numpy as np


class Benchmarks:

    def quadratic_sum(self, X):
        """ Quadratic sum problem """
        s = 0
        for i, x in enumerate(X):
            s += x ** 2
        return s

    def quadratic_sum_scaled(self, X):
        """ Scaled quadratic sum problem """
        s = 0
        for i, x in enumerate(X):
            s += (i + 1) ** 2 * x ** 2
        return s

    def schwefel(self, X):
        """
        Schwefel's problem
        """
        s1 = 0
        s2 = 0
        n = len(X)
        for i, x in enumerate(X):
            s2 = 0
            for j in range(i, n):
                s2 += x ** 2
        s1 += s2 ** 2
        return s1

    def staircase(self, X):
        """
        Staircase function problem
        """
        s = 0
        for i, x in enumerate(X):
            s += math.floor(x ** 2)

        return s

    def rastrigen(self, X):
        """
        Rastrigen function problem
        """

        n = len(X)
        s = 0
        for i, x in enumerate(X):
            s = x ** 2 * math.cos( 2 * math.pi * x)

        s +=  10 * n
        return s

    def func11(self, x, y):
        """
        Function 1 from the first exercise sheet (bivariate)
        """
        return 2 * x ** 2 - 3 * x * y + 2 * y ** 2 + 6 * y

    def func12(self, x, y):
        """
        Function 2 from the first exercise sheet (bivariate)
        """
        return 4 * x ** 2 + 4 * x * y + 2 * y ** 2 + 24 * x - 4 * y + 5

    def func21(self, x):
        """
         Function 1 from the first exercise sheet (univariate)
        """
        return (x + 3) ** 2

    def func22(self, x):
        """
        Function 2 from the first exercise sheet (univariate)
        """
        return x ** 2 - 5 * np.cos(10 * x)

    def func23(self, x, y):
        """
        Function 3 from the first exercise sheet (bivariate)
        """
        return x ** 2 + y ** 2

    def func24(self, x, y):
        """
        Function 4 from the first exercise sheet (bivariate)
        """
        return x * np.sin(x) + 3 * y ** 2

    def func31(self, x):
        """
        Function 1 from the third exercise
        """
        return x ** 2 - 100 * np.cos(10 * x)

    def func32(self, x, y):
        """
        Function 2 from the third exercise
        """
        return x ** 2 + y ** 2

    def func41(self, x, y):
        """
        Function 1 from the fourth exercise
        """
        return x ** 3 - y ** 3 + y ** 2 + 1000 * np.cos(x) * np.sin(y)

    def gen_espaced(self, nvars, min, max, n):
        """
        Generate evenly spaced data
        """
        D = np.zeros((nvars, n))
        for i in range(0, nvars):
            D[i,] = np.linspace(min, max, n)
        return D

    def gen_normal(self, nvars, mu, sigma, n):
        """
        Generate normal distributed date
        """
        D = np.zeros((nvars, n))
        for i in range(0, nvars):
            D[i,] = np.random.normal(mu, sigma, n)
        return D
