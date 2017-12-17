import numpy as np
import matplotlib.pyplot as plt
from utils import teachers_variables, run_model, simulated_annealing
import pickle

plt.switch_backend('agg')

N = 1000
initial_beta = 0.05
alpha_max = 5
beta_pace = 100
beta_increase = 10
alphas = [1, 2, 3, 4, 5]
max_numb_states = 20000

print("Generating variables")
w_t, x_t, y_t = teachers_variables(n=N, alpha_max=alpha_max)
print("Done")

energies = {}
overlaps = {}


for alpha in alphas:
    print("Trying beta0 = ", initial_beta, ",  beta_increase = ", beta_increase)
    e,o = simulated_annealing(x_t, y_t, w_t, alpha, initial_beta, beta_pace, beta_increase, N, 
                          max_numb_states=max_numb_states, num_repetitions=10, verbose=False)
    energies[alpha] = e[-1]/int(alpha*N)
    overlaps[alpha] = o
    plt.plot(np.arange(max_numb_states+1), e)
    plt.savefig("SA_alpha"+str(alpha)+".png")
    plt.clf()

with open("SA_energies.pickle", 'wb') as f:
    pickle.dump(energies, f)
with open("SA_overlaps.pickle", 'wb')  as f:
    pickle.dump(overlaps, f) 