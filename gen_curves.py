import numpy as np
import pickle
import matplotlib.pyplot as plt

def ener_over_alphas():
    """
    Evolution of the final normalized energy and overlap for different values of alpha, with hyperparameters (beta_initial=0.05, beta_pace=10)
    """
    with open("SA_energies_alphas.pickle",'rb') as f:
        energies = pickle.load(f)
    with open("SA_overlaps_alphas.pickle",'rb') as f:
        overlaps = pickle.load(f)

    alphas = list(energies.keys())
    e = [energies[a] for a in alphas]
    o = [overlaps[a] for a in alphas]
    plt.plot(alphas, e, color="red", label="normalized energy")
    plt.plot(alphas, o, color="blue", label="overlap")
    plt.ylim(0,1)
    plt.legend(loc='upper left')
    plt.xlabel("alpha")

    plt.show(block=False)

def ener_beta_pace():
    """
    Evolution of the normalized energy in function of beta_pace for fixed alpha=1, fixed beta_initial=0.05
    """
    with open("SA_energies0.05.pickle",'rb') as f:
        energies = pickle.load(f)

    beta_pace = sorted(list(energies.keys()))
    e = [energies[b] for b in beta_pace]
    plt.plot(beta_pace, e, color="blue")
    plt.xlabel("beta_pace")
    plt.ylabel("normalized energy")

    plt.show(block=False)