"""
Recursive implementation of the fibonacci sequence calculation by using
an outer/inner function architecture.

Author:         Roman Kalkreuth, roman.kalkreuth@tu-dortmund.de
                https://orcid.org/0000-0003-1449-5131
                https://ls11-www.cs.tu-dortmund.de/staff/kalkreuth
                https://twitter.com/RomanKalkreuth
"""

def fibonacci(n):
    # Validate the input
    if not isinstance(n, int):
        raise TypeError("Number of must be an integer!")
    if n <= 1:
        raise ValueError("Number must be greater than one!")
    fib = []

    # Inner function to calculate the fibonacci sequence
    # a represents n-1 and b n-2
    def calc_fib(fib, n):
        l = len(fib)
        i = l - 1
        if l == n:
            # If we have reached n, end the recursion
            return fib
        elif l <= 1:
            # The first two number are 1
            fib.append(1)
        else:
            # Calculate the i-th number with the previous numbers
            a = fib[i]
            b = fib[i - 1]
            fib.append(a + b)
            # Recursion step
        return calc_fib(fib, n)

    # Call the inner function and start the recursion
    return calc_fib(fib, n)


fib = fibonacci(20)

print(fib)
