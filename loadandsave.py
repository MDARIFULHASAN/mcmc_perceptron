import matplotlib.pyplot as plt
import pickle

energies = []
overlaps = []
alphas = [1, 2, 3, 4, 5]

with open("SA_energies.pickle", 'rb') as f:
    energies = pickle.load(f)
with open("SA_overlaps.pickle", 'rb')  as f:
    overlaps = pickle.load(f)

plt.plot(alphas, energies, color="red")
plt.plot(alphas, overlaps, color="blue")

plt.savefig("simulated_annealing.png")