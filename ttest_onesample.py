"""
Performs a one-sample t-test and prints out the test statistic as well as the
critical value(s).

Author:         Roman Kalkreuth, roman.kalkreuth@tu-dortmund.de
                https://orcid.org/0000-0003-1449-5131
                https://ls11-www.cs.tu-dortmund.de/staff/kalkreuth
                https://twitter.com/RomanKalkreuth
"""

import math

import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats
import seaborn as sns


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


def p_value(tval, df, type):
    p = stats.t.sf(abs(tval), df=df)
    if (type == "two"):
        p = p * 2
    return p


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


def density_plot(size, c, t):
    """
    Creates a density plot using the seaborn package
    """
    tvals = stats.t.rvs(df=df, size=size)
    density = sns.kdeplot(tvals, shade=True)
    density.axvline(c, color='r')
    density.axvline(-1 * c, color='r')
    density.axvline(t, color='g')
    plt.show()


# Sample size, mean and standard deviation for normal distributed data
mu_norm, sigma = 0.0, 5.0
n = 100

# Test values: level of significance, hypothesized mean and test type
mu_hyp = 0.1
alpha = 0.1
type = "two"

# Sample n normal distributed random values
sample = gen_normal(mu_norm, sigma, n)

# Perform the one sample t-test
mean = avg(sample)
mu_hyp = mean + 0.02
diff = difference(mean, mu_hyp)
std = stdev(sample, mean)
error = sem(mean, n)
df = dof(n)
t = test_statistic(diff, error)

# Determine the critical value(s)
c = critical_value(alpha, df, type)

p = p_value(t, df, type)

print("t: " + str(t))

if type == "two":
    print("c: +/- " + str(c ** 2))
else:
    print("c: " + str(c))

print("p: " + str(p))

# Create a density plot
density_plot(100000, c, t)

