# library imports
import numpy as np

# our imports
from hw_functions import integrator, helper_functions


############################# problem 4 #############################
inertia_tensor = np.array([[ 100.0, 0.0, 0.0 ],
                           [ 0.0, 75.0, 0.0 ],
                           [ 0.0, 0.0, 80.0 ]])

k = 5 # Nm
p = np.array([[ 10.0, 0.0, 0.0 ],
              [ 0.0, 10.0, 0.0 ],
              [ 0.0, 0.0, 10.0 ]])


w_b_n_0 = np.array([ 30.0, 10.0, -20.0 ]) * np.pi / 180
x_0 = np.concatenate([np.array([ 0.1, 0.2, -0.1 ]), w_b_n_0])

def get_control_n(x_n):

    mrp_n = x_n[:3]
    att_rate_n = x_n[3:6]

    att_rate_n_tilde = helper_functions.get_tilde_matrix(att_rate_n)

    return -1 * k * mrp_n - p @ att_rate_n + att_rate_n_tilde @ inertia_tensor @ att_rate_n

def get_state_dot(x_n, t):

    mrp_n = x_n[:3]
    att_rate_n = x_n[3:6]

    mrp_n_norm = np.linalg.norm(mrp_n)
    mrp_n_tilde = helper_functions.get_tilde_matrix(mrp_n)

    att_rate_n_tilde = helper_functions.get_tilde_matrix(att_rate_n)

    mrp_dot = ( 1/4 ) * ( ( 1 - mrp_n_norm**2 ) * np.eye(3) + 2 * mrp_n_tilde + 2 * mrp_n_norm**2 ) @ att_rate_n
    acc_rate = np.linalg.inv(inertia_tensor) @ ( -1 * att_rate_n_tilde @ inertia_tensor @ att_rate_n + get_control_n(x_n) )

    return np.concatenate( [mrp_dot, acc_rate] )

mrp_summary = integrator.runge_kutta(get_state_dot, x_0, 0, 120, is_mrp=True)
print(np.linalg.norm(mrp_summary[3000][:3]))


############################# problem 5 #############################