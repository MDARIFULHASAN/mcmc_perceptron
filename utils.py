import numpy as np


def rand_unit():
    n = np.random.random_sample()
    if n > 0.5:
        return 1
    else:
        return -1


def teachers_variables(n, alpha_max):
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


def simulated_annealing(x_t, y_t, w_t, alpha, initial_beta, beta_pace, beta_increase, n, max_numb_states=100,
                        num_repetitions=10, verbose=False):
    energies = []
    overlaps = []
    beta = initial_beta
    m = int(alpha*n)

    for i in range(num_repetitions):
        if i % beta_pace == 0:
            beta *= beta_increase

        w, e = metro_chain(alpha=alpha, beta=beta, x_t=x_t, y_t=y_t, n=n, max_numb_states=max_numb_states,
                           verbose=verbose)
        energies.append(e)
        overlaps.append(overlap(w, w_t, n))

    mean_energy = get_mean_energy(energies, num_repetitions)
    mean_overlap = np.array(overlaps).mean()
    return mean_energy, mean_overlap
