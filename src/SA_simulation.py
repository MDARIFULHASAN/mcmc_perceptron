"""
This script computes the normalized energy and overlap for different values of alpha for the simulated annealing method
"""

import numpy as np
import matplotlib.pyplot as plt
from utils import teachers_variables, run_model, run_SA
import pickle

plt.switch_backend('agg')

N = 1000
initial_beta = 0.05
alpha_max = 5
beta_pace = 500
beta_increase = 1.2
alphas = [1, 2, 3, 4, 5]
max_numb_states = 10000

print("Generating variables")
w_t, x_t, y_t = teachers_variables(n=N, alpha_max=alpha_max)
print("Done")

energies = {}
overlaps = {}


for alpha in alphas:
    print("Trying alpha= ", alpha)
    e, o = run_SA(x_t, y_t, w_t, alpha, initial_beta, beta_pace, beta_increase, N,
                  max_numb_states=max_numb_states, num_repetitions=10, verbose=True)
    energies[alpha] = e[-1]/int(alpha*N)
    overlaps[alpha] = o
    plt.plot(np.arange(max_numb_states+1), e)
    plt.savefig("SA_alpha"+str(alpha)+".png")
    plt.clf()

with open("pickles/SA_energies.pickle", 'wb') as f:
    pickle.dump(energies, f)
with open("pickles/SA_overlaps.pickle", 'wb') as f:
    pickle.dump(overlaps, f) 