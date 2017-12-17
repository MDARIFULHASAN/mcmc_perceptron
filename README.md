# Markov Chains course project

Group:
    * Adrian Valente
    * Armand Boschin
    * Quentin Rebjock

This project is about the use of MCMC and simulated annealing for the Ising perceptron.

## Implementations of the algorithms
Our implementation of the MCMC algorithm and of the simulated annealing can be found in the file utils.py. Most notably the metro_chain and SA_chain are the most important functions that actually execute the algorithms. The docstrings are provided in the code.

## Questions and curves’ generation
In order the generate the curves to answer question 1, we used the iPython notebook markov_project.ipynb.
In order to answer questions 2 and 3, we used the script Question2-3.py
For the simulated annealing part, we used a script SA_crossval.py to try different values for the hyperparameters, and SA_simulation.py to try different values of alpha.