"""
This script executes a cross-validation of parameters initial_beta and beta_increase for the simulated annealing method.
Modify the lists initial_betas and beta_increases to try different values
"""

import numpy as np
import matplotlib.pyplot as plt
from utils import teachers_variables, run_model, run_SA
import pickle

plt.switch_backend('agg')

N = 1000
initial_betas = [0.01, 0.05]
alpha_max = 5
beta_pace = 500
beta_increases = [1.1, 1.2, 1.4, 1.8, 2.5, 5]
alpha = 1
max_numb_states = 10000

print("Generating variables")
w_t, x_t, y_t = teachers_variables(n=N, alpha_max=alpha_max)
print("Done")

energies = {}
overlaps = {}


for initial_beta in initial_betas:
    for beta_increase in beta_increases:
        print("Trying beta0 = ", initial_beta, ",  beta_increase = ", beta_increase)
        e, o = run_SA(x_t, y_t, w_t, alpha, initial_beta, beta_pace, beta_increase, N,
                      max_numb_states=max_numb_states, num_repetitions=10, verbose=True)
        energies[(initial_beta, beta_increase)] = e[-1]/int(alpha*N)
        overlaps[(initial_beta, beta_increase)] = o
        plt.plot(np.arange(max_numb_states+1), e)
        plt.savefig("SA_"+str(initial_beta)+"_"+str(beta_increase)+".png")
        plt.clf()

with open("pickles/SA_energies.pickle", 'wb') as f:
    pickle.dump(energies, f)
with open("pickles/SA_overlaps.pickle", 'wb') as f:
    pickle.dump(overlaps, f) 