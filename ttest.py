"""
Performs a one-sample t-test and prints out the test statistic as well as the
critical value(s)
"""

import math
import numpy as np
import scipy.stats as stats


def difference(mean, mu):
    """ Difference between the sample mean and hypothesized mean"""
    return mean - mu


def avg(sample):
    """ Mean of the sample """
    return sum(sample) / len(sample)


def stdev(sample, mean):
    """ Calculates the standard deviation manually """
    sum = 0
    for s in sample:
        sum += math.pow(s - mean, 2)
    var = sum / len(sample)
    std = math.sqrt(var)
    return std


def sem(stdev, n):
    """ Standard error of mean of the sample """
    return stdev / math.sqrt(n)


def dof(n):
    """ Degrees of freedom  """
    return n - 1


def test_statistic(diff, error):
    """ Test statistic t """
    return diff / error


def gen_normal(mu, sigma, n):
    """
    Generate a sample of normal distributed date
    """
    return np.random.normal(mu, sigma, n)


def critical_value(alpha, df, type):
    """ Critical value(s) for the given level of
    significance and the degrees of freedoms """
    if type == "left":
        q = alpha
    elif type == "right":
        q = 1.0 - alpha
    elif type == "two":
        q = 1.0 - alpha / 2.0
    return stats.t.ppf(q, df)


# Sample size, mean and standard deviation for normal distributed data
mu = 0
n = 100
sigma = 1

# Test values: level of significance, hypothesized mean and test type
mu = 0.5
alpha = 0.05
type = "two"

# Sample n normal distributed random values
sample = gen_normal(mu, sigma, n)

# Perform the one sample t-test
mean = avg(sample)
diff = difference(mean, mu)
std = stdev(sample, mean)
error = sem(mean, n)
df = dof(n)
t = test_statistic(diff, error)

# Determine the critical value(s)
c = critical_value(alpha, df, type)

print("t: " + str(t))

if (type == "two"):
    print("c: +/- " + str(c ** 2))
else:
    print("c: " + str(c))
