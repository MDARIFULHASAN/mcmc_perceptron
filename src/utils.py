import numpy as np
import scipy.io

def read_matlab(filename):
    var = scipy.io.loadmat(filename)
    y = var['y']
    X = var['X']
    M = int(var['M'])
    N = int(var['N'])
    M_test = int(var['M_test'])
    X_test = var['X_test']
    return y,X,M,N,M_test,X_test



def rand_unit():
    n = np.random.random_sample()
    if n > 0.5:
        return 1
    else:
        return -1


def teachers_variables(n, alpha_max):
    """
    Generate a random w, and a corresponding set of values x and y
    """
    m_max = alpha_max * n
    w_t = [rand_unit() for _ in range(n)]
    x_t = np.random.normal(size=(m_max, n))
    y_t = [int(np.sign(x_t[i].dot(w_t))) for i in range(m_max)]
    return w_t, x_t, y_t


def compute_energy(w, x_t, y_t, m):
    """
    Compute energy relative to the weight vector of length M in extracted from W
    """
    v = np.sign(np.dot(x_t[:m], w)) - y_t[:m]
    return np.vdot(v, v)/2


def metro_chain(x_t, y_t, alpha, beta, n, max_numb_states=None, energy_stop=None, verbose=False):
    """
    This function executes the MCMA algorithm
    :param x_t: the random variables x (list of numpy arrays)
    :param y_t: the corresponding y  (list)
    :param alpha: the alpha value
    :param beta: the beta value
    :param n: the number of coordinates
    :param max_numb_states: the maximal number of iterations (if None, the criterion energy_stop should be provided)
    :param energy_stop: a stopping threshold for the energy (if None, max_numb_states should be provided)
    :param verbose:
    :return: the final w obtained, a time series of the energy (in list format)
    """
    m = int(alpha * n)

    # initialization
    current_w = [rand_unit() for _ in range(n)]
    current_energy = compute_energy(current_w, x_t, y_t, m)

    energy = [current_energy]
    counter = 0

    if max_numb_states is None:
        max_numb_states = np.inf
    if energy_stop is None:
        energy_stop = -1

    while current_energy > energy_stop and counter < max_numb_states:
        counter += 1
        current_energy = compute_energy(current_w, x_t, y_t, m)

        if verbose and counter % 100 == 0:
            print('Iteration {} and energy : {}'.format(counter, current_energy))

        # pick coordinate at random
        coord = np.random.randint(0, high=n)

        # flip this coordinate
        new_w = current_w.copy()
        new_w[coord] = -new_w[coord]

        # decide if new state is accepted
        new_energy = compute_energy(new_w, x_t, y_t, m)

        if new_energy < current_energy:
            current_w = new_w
            energy.append(new_energy)
        else:
            a = np.exp(- beta * (new_energy - current_energy))
            r = np.random.random_sample()
            if r < a:
                current_w = new_w
                energy.append(new_energy)
            else:
                energy.append(current_energy)

    return current_w, energy


def get_mean_energy(energies, num_repetitions=10):
    """
    Computes the mean of several time series
    :param energies: the time series, as a list of lists
    :param num_repetitions: the number of time series
    :return: a list representing the mean of the time series
    """
    n = np.max([len(i) for i in energies])
    energy = []
    for i in range(n):
        tmp = 0
        counter = 0
        for k in range(num_repetitions):
            if i < len(energies[k]):
                tmp += energies[k][i]
            counter += 1

        energy.append(tmp / counter)
    return energy


def run_model(alpha, beta, n, x_t, y_t, max_numb_states=100, num_repetitions=10, verbose=False):
    """
    Run num_repetitions of the MCMC model
    :return: a list representing the mean time series of the energy
    """
    energies = []
    for _ in range(num_repetitions):
        _, tmp = metro_chain(alpha=alpha, beta=beta, x_t=x_t, y_t=y_t, n=n, max_numb_states=max_numb_states,
                             verbose=verbose)
        energies.append(tmp)
        if verbose:
            print('Number of states in the chain :{}'.format(len(tmp)))
    energy = get_mean_energy(energies, num_repetitions=num_repetitions)
    return energy


def overlap(w, w_t, n):
    return np.vdot(w, w_t)/n


def SA_chain(alpha, initial_beta, beta_pace, beta_increase, x_t, y_t, n, max_numb_states):
    """
    Do an iteration of simulated annealing
    :param alpha: the alpha value
    :param initial_beta: the initial value of beta
    :param beta_pace: the frequency of updates to the value of beta (in number of iterations)
    :param beta_increase: the multiplicator of beta, for the updates
    :param x_t: random x values (list of numpy arrays)
    :param y_t: corresponding y values (list)
    :param n: number of coordinates
    :param max_numb_states: the maximal number of iterations
    :return: the final w obtained, and the energy time series (as a list)
    """
    m = int(alpha * n)

    # initialization
    current_w = [rand_unit() for _ in range(n)]
    current_energy = compute_energy(current_w, x_t, y_t, m)

    energy = [current_energy]
    counter = 0

    beta = initial_beta

    while counter < max_numb_states:
        counter += 1
        if counter % beta_pace == 0:
            beta *= beta_increase
            print(counter)

        # pick coordinate at random
        coord = np.random.randint(0, high=n)

        # flip this coordinate
        new_w = current_w.copy()
        new_w[coord] = -new_w[coord]

        # decide if new state is accepted
        new_energy = compute_energy(new_w, x_t, y_t, m)

        if new_energy < current_energy:
            current_w = new_w
            energy.append(new_energy)
            current_energy = new_energy
        else:
            a = np.exp(- beta * (new_energy - current_energy))
            r = np.random.random_sample()
            if r < a:
                current_w = new_w
                energy.append(new_energy)
                current_energy = new_energy
            else:
                energy.append(current_energy)

    return current_w, energy


def run_SA(x_t, y_t, w_t, alpha, initial_beta, beta_pace, beta_increase, n, max_numb_states=100,
           num_repetitions=10):
    """
    Repeat num_repetitions experiments of simulated annealing
    :return: a list representing the mean time series of the energy, and the mean final overlap
    """
    energies = []
    overlaps = []

    for i in range(num_repetitions):
        w, e = SA_chain(alpha=alpha, initial_beta=initial_beta, beta_pace=beta_pace, beta_increase=beta_increase,
                        x_t=x_t, y_t=y_t, n=n, max_numb_states=max_numb_states)
        energies.append(e)
        overlaps.append(overlap(w, w_t, n))

    mean_energy = get_mean_energy(energies, num_repetitions)
    mean_overlap = np.array(overlaps).mean()
    return mean_energy, mean_overlap