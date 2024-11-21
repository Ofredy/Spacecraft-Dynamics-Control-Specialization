# library imports
import numpy as np

# our imports
from hw_functions import integrator, helper_functions, mrp_functions


############################# problem 4 #############################
inertia_tensor = np.array([[ 100.0, 0.0, 0.0 ],
                           [ 0.0, 75.0, 0.0 ],
                           [ 0.0, 0.0, 80.0 ]])

k = 5 # Nm
p = np.array([[ 10.0, 0.0, 0.0 ],
              [ 0.0, 10.0, 0.0 ],
              [ 0.0, 0.0, 10.0 ]])


w_0 = np.array([ 30.0, 10.0, -20.0 ]) * np.pi / 180
x_0 = np.concatenate([np.array([ 0.1, 0.2, -0.1 ]), w_0])

def get_att_reg_control_n(x_n):

    mrp_n = x_n[:3]
    w_n = x_n[3:6]

    return -1 * k * mrp_n - p @ w_n 

def get_att_reg_state_dot(x_n, t):

    mrp_n = x_n[:3]
    w_n = x_n[3:6]

    mrp_n_norm = np.linalg.norm(mrp_n)
    mrp_n_tilde = helper_functions.get_tilde_matrix(mrp_n)

    mrp_dot = ( 1/4 ) * ( ( 1 - mrp_n_norm**2 ) * np.eye(3) + 2 * mrp_n_tilde + 2 * np.outer(mrp_n, mrp_n) ) @ w_n
    w_dot = np.linalg.inv(inertia_tensor) @ ( get_att_reg_control_n(x_n) + np.array([ 0.5, -0.3, 0.2 ]) ) 

    return np.concatenate( [mrp_dot, w_dot] )

mrp_sum = integrator.runge_kutta(get_att_reg_state_dot, x_0, 0, 120, is_mrp=True)
print(np.linalg.norm(mrp_sum[3500][:3]))