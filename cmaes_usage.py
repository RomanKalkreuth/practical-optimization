import numpy as np
import cmaes as cma_es
import benchmarks as bm

# Meta parameters
dim = 2
sigma = 1
generations = 1000
pop_size = 128
mean = np.zeros(dim)

# Init the cma es optimizer
optimizer = cma_es.CMA(mean=mean, sigma=sigma, population_size=pop_size)

# Benchmark function selection
benchmarks = bm.Benchmarks()
func = benchmarks.func61

# Iterate over the number of generations
for gen in range(generations):
    sol = []
    # Iterate over the population size
    for i in range(pop_size):
        x = optimizer.ask()
        y = func(x)
        sol.append((x, y))
        # output of the parameters and function value
        #print("Generation: " + str(gen+1) + " Function value" + str(y) + "Function parameter: x1=" + str(x[0])
        #      + ", x2 =" + str(x[1]))
    optimizer.tell(sol)

