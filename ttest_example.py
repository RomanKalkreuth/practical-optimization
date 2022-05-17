"""

Author:         Roman Kalkreuth, roman.kalkreuth@tu-dortmund.de
                https://orcid.org/0000-0003-1449-5131
                https://ls11-www.cs.tu-dortmund.de/staff/kalkreuth
                https://twitter.com/RomanKalkreuth
"""

import numpy as np
import scipy.stats as stats

# Two samples of different dimensions
d1 = np.array([22.0, 16.0, 21.7, 21.0, 30.0, 26.0, 12.0, 23.2, 28.0, 23.0])
d2 = np.array([13.3, 6.0, 20.0, 8.0, 14.0, 19.0, 18.0, 25.0, 16.0, 24.0, 15.0, 1.0, 15.0])

# Two random samples with normally distributed values (used as counter example)
#d1 = np.random.normal(0,4, 10)
#d2 = np.random.normal(0,4, 13)


# Get the dimension of the samples
n1 = len(d1)
n2 = len(d2)

# Calculate the mean values
x1 = np.average(d1)
x2 = np.average(d2)

# Calculate the absolute difference
xdiff = abs(x1 - x2)

# Get the standard deviation of the two samples
s1 = np.std(d1)
s2 = np.std(d2)

# Calculate the degrees of freedom
df = n1 + n2 - 2

# Calculate the pooled variance for both samples
sp2 = (((n1 - 1) * s1 ** 2) + ((n2 - 1) * s2 ** 2)) / df

# Calculate the pooled standard deviation
sp = np.sqrt(sp2)

# Calculate the standard error
sed = sp * np.sqrt(1/n1 + 1/n2)

# Determine the test statistic t
t = xdiff/sed

# Get the p values from the t distribution with 'df' degrees of freedom
p = stats.t.sf(abs(t), df=df)

# For a two-tailed test multiply the p value by two
p *= 2

print("n1: " + str(n1))
print("n2: " + str(n2) + "\n")
print("x1: " + str(x1))
print("x2: " + str(x2) + "\n")
print("s1: " + str(s1))
print("s2: " + str(s2) + "\n")
print("sp2: " + str(sp2))
print("sed: " + str(sed)+ "\n")
print("t: " + str(t))
print("p: " + str(p))