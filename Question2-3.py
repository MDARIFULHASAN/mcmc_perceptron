"""
This script computes the value of the normalized energy and final overlap for the MCMC method, for several
values of alpha and beta. Modify the lists betas and alphas to choose values.
"""

import numpy as np
import matplotlib.pyplot as plt
import pickle
from utils import teachers_variables, run_model, metro_chain, compute_energy

plt.switch_backend('agg')

n = 1000
alpha_max = 5
print("Generating variables...")
w_t, x_t, y_t = teachers_variables(n=n, alpha_max=alpha_max)

betas = [0.05, 0.5, 1, 5]
T = 500
alphas = np.arange(1, 5)
max_numb_states = 10000

energies = {}
overlaps = {}

def normalized_energy_overlap(x_t, y_t, alpha, beta, n, max_numb_states, num_repetitions=5):
    e = 0
    o = 0
    for i in range(num_repetitions):
        w, _ = metro_chain(x_t=x_t, y_t=y_t, alpha=alpha, beta=beta, n=n, max_numb_states=max_numb_state)
        e += compute_energy(w, x_t, y_t, alpha*n)/(alpha*n)
        o += np.vdot(w, w_t)/n
    return e/num_repetitions, o/num_repetitions

for beta in betas:
    for alpha in alphas:
        print("alpha = "+str(alpha)+"  beta = "+str(beta))
        e, o = normalized_energy_overlap(x_t, y_t, alpha, beta, n, max_numb_states)
        energies[(alpha,beta)] = e
        overlaps[(alpha,beta)] = o

with open('q2.pickle', 'wb') as f:
    pickle.dump(energies, f)

with open('q3.pickle', 'wb') as f:
    pickle.dump(overlaps, f)