"""
Benchmark class that contains the benchmarking functions
for the exercises
"""

import numpy as np
import math

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

    def staircase(self, X):
        """ Staircase function problem """
        s = 0
        for i, x in enumerate(X):
            s += math.floor(x ** 2)

        return s

    #def rastrigen(self, X):
    #
    #   return X

    def func11(self,x, y):
        """
        Function 1 from the first exercise sheet (bivariate)
        """
        return 2 * x ** 2 - 3 * x * y + 2 * y ** 2 + 6 * y

    def func12(self,x, y):
        """
        Function 2 from the first exercise sheet (bivariate)
        """
        return 4 * x ** 2 + 4 * x * y + 2 * y ** 2 + 24 * x - 4 * y + 5

    def func21(self,x):
        """
         Function 1 from the first exercise sheet (univariate)
        """
        return (x + 3) ** 2

    def func22(self,x):
        """
        Function 2 from the first exercise sheet (univariate)
        """
        return x ** 2 - 5 * np.cos(10 * x)

    def func23(self,x, y):
        """
        Function 3 from the first exercise sheet (bivariate)
        """
        return x ** 2 + y ** 2

    def func24(self,x, y):
        """
        Function 4 from the first exercise sheet (bivariate)
        """
        return x * np.sin(x) + 3 * y ** 2

    def func31(self,x):
        """
        Function 1 from the third exercise
        """
        return x ** 2 - 100 * np.cos(10 * x)

    def func32(self,x,y):
        """
        Function 2 from the third exercise
        """
        return x ** 2 + y ** 2

    def gen_espaced(self,vars, min, max, n):
        """
        Generate evenly spaced data
        """
        D = np.zeros((vars, n))
        for i in range(0, vars):
            D[i,] = np.linspace(min, max, n)
        return D

    def gen_normal(self,vars, mu, sigma, n):
        """
        Generate normal distributed date
        """
        D = np.zeros((vars, n))
        for i in range(0, vars):
            D[i,] = np.random.normal(mu, sigma, n)
        return D
