"""
Creates a qqplot using the statsmodels package
"""

import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt

np.random.seed(0)

mean, sigma, size = 0, 1, 100
low, high = 0, 1

# Sample normal and uniform distribured values
dn = np.random.normal(mean,sigma, size)
du = np.random.uniform(low,high, size)

# Create qqplot
fig = sm.qqplot(du, line='45')
plt.show()