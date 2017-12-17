import numpy as np
from utils import teachers_variables, rand_unit, compute_energy

N = 1000
max_numb_states = 20000
alpha_max = 1

print("Generating variables")
w_t, x_t, y_t = teachers_variables(n=N, alpha_max=alpha_max)
print("Done")


def try_random(x_t, y_t, w_t, n, max_numb_states):
    e = compute_energy([1]*n, x_t, y_t, n)
    print(e)
    argmin = [1]*n
    for i in range(max_numb_states):
        w = [rand_unit() for _ in range(n)]
        tmp = compute_energy([0]*n, x_t, y_t, n)
        if tmp < e:
            e = tmp
            argmin = w
            print(e)
    return argmin, e

w, e = try_random(x_t, y_t, w_t, N, max_numb_states)
print(e)