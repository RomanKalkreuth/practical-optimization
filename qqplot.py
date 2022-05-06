"""
Creates a qqplot using the statsmodels package.

Author:         Roman Kalkreuth, roman.kalkreuth@tu-dortmund.de
                https://orcid.org/0000-0003-1449-5131
                https://ls11-www.cs.tu-dortmund.de/staff/kalkreuth
                https://twitter.com/RomanKalkreuth
"""

import matplotlib.pyplot as plt
import numpy as np
import statsmodels.api as sm

np.random.seed(0)

mean, sigma, size = 0, 1, 100
low, high = 0, 1

# Sample normal and uniform distributed values
dn = np.random.normal(mean, sigma, size)
du = np.random.uniform(low, high, size)

# Create the qqplot
fig = sm.qqplot(du, line='45')
plt.show()
