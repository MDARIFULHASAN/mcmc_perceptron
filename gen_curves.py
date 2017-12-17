import numpy as np
import pickle
import matplotlib.pyplot as plt

def q2():
    """
    Answer to questions 2 and 3 (values of the energy and the overlap for different values of alpha)
    :return:
    """
    with open("q2.pickle", 'rb') as f:
        energies = pickle.load(f)
    with open("q3.pickle", 'rb') as f:
        overlaps = pickle.load(f)
    betas = [0.05, 0.5, 1, 5]
    alphas = list(range(1,5))

    for beta in betas:
        e = [energies[(a,beta)] for a in alphas]
        o = [overlaps[(a,beta)] for a in alphas]
        plt.plot(alphas, e, color="red", label="normalized energy")
        plt.plot(alphas, o, color="blue", label="overlap")

        plt.ylim(0,1.2)
        plt.legend(loc='upper left')
        plt.xlabel("alpha")
        plt.title("beta = "+str(beta))
        plt.savefig("q2-"+str(beta)+".png")
        plt.clf()

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

def cross_val_curve():
    """
    Energy as a function of beta_increase, for different values of beta_initial.
    """
    with open("SA_energies_crossval.pickle",'rb') as f:
        energies = pickle.load(f)

    initial_betas = [0.01, 0.05]
    beta_increases = [1.1, 1.2, 1.4, 1.8, 2.5, 5]
    for beta in initial_betas:
        e = [energies[(beta,b)] for b in beta_increases]
        plt.plot(beta_increases, e, label="initial_beta="+str(beta))

    plt.xlabel("beta_increases")
    plt.ylabel("normalized energy")
    plt.ylim(0.4,1)
    plt.legend(loc='upper right')
    plt.show(block=False)
