"""
Plots the density of normal and t distributed values.

Author:         Roman Kalkreuth, roman.kalkreuth@tu-dortmund.de
                https://orcid.org/0000-0003-1449-5131
                https://ls11-www.cs.tu-dortmund.de/staff/kalkreuth
                https://twitter.com/RomanKalkreuth
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as stats
import seaborn as sns


def density_plot(size, df):
    """
    Creates a density plot using the seaborn package
    """

    # Sample the normal and t distributed values
    nvals = np.random.normal(0, 1, size)
    tvals = stats.t.rvs(loc=0, scale=1, df=df, size=size)

    # Create a pandas dataframe
    df = pd.DataFrame({"normal distribution": nvals,
                       "t-distribution": tvals})

    # Plot the dataframe
    sns.kdeplot(data=df, shade=True)
    plt.show()


# Set the seed
np.random.seed(0)

# Degrees of freedom and sample size
df = 5
size = 10000

density_plot(size, df)
