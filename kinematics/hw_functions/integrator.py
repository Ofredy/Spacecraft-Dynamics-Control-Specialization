import numpy as np


def integrator(func_x, x_0, t_0, t_f, dt=0.1, norm_value=False):

    state_summary = []

    x_n = x_0
    state_summary.append(x_n)
    t = t_0

    while t < t_f:

        f = func_x(x_n, t)

        x_n = x_n + f * dt

        if norm_value:
            x_n = x_n / np.linalg.norm(x_n)

        state_summary.append(x_n)

        t += dt

    return state_summary
