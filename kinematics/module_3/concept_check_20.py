import numpy as np

from hw_functions import integrator, helper_functions

######## problem 4 ########
def get_mrp_p_t(mrp_n, t):

    w_t = np.array([ np.sin(0.1*t), 0.01, np.cos(0.1*t) ]) * 20 * np.pi / 180

    mrp_norm = np.linalg.norm(mrp_n)
    mrp_tilde = helper_functions.get_tilde_matrix(mrp_n)

    return (1/4) * ( ( 1 - mrp_norm**2 ) * np.eye(3) + 2 * mrp_tilde + 2 * np.dot(mrp_n, np.transpose(mrp_n)) ) @ w_t

mrp_summary = integrator.runge_kutta(get_mrp_p_t, np.array([0.4, 0.2, -0.1]), 0, 42, dt=0.01, is_mrp=True)
print(np.linalg.norm(mrp_summary[4200]))