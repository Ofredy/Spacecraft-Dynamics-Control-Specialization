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
    w_dot = np.linalg.inv(inertia_tensor) @ get_att_reg_control_n(x_n) 

    return np.concatenate( [mrp_dot, w_dot] )

mrp_sum = integrator.runge_kutta(get_att_reg_state_dot, x_0, 0, 120, is_mrp=True)
print(np.linalg.norm(mrp_sum[3000][:3]))


############################# problem 5 #############################

# using k to denote timestep for this problem
dt = 0.01
f = 0.05

mrp_b_n_0 = np.array([ 0.1, 0.2, -0.1 ])
w_b_n_0 = np.array([ 30.0, 10.0, -20.0 ]) * np.pi / 180

x_0 = np.concatenate([mrp_b_n_0, w_b_n_0])

def get_target_mrp_r_n(t):
    
    return np.array([ 0.2 * np.sin( f*t ), 0.3 * np.cos( f*t ), -0.3 * np.sin( f*t ) ])

def get_target_mrp_r_n_dot(t):

    return np.array([ 0.2 * f * np.cos( f*t ), -0.3 * f * np.sin( f*t ), -0.3 * f * np.cos( f*t ) ])

def get_target_w_r_n(t):

    # solving for mrp_r_n_k
    mrp_r_n_k = get_target_mrp_r_n(t)

    # solving for w_r_n
    mrp_r_n_dot_k = get_target_mrp_r_n_dot(t)

    mrp_r_n_norm = np.linalg.norm(mrp_r_n_k)
    mrp_r_n_tilde = helper_functions.get_tilde_matrix(mrp_r_n_k)

    return 4 * np.linalg.inv( ( 1 - mrp_r_n_norm**2 ) * np.eye(3) + 2 * mrp_r_n_tilde + 2 * np.outer(mrp_r_n_k, mrp_r_n_k) ) @ mrp_r_n_dot_k

def get_target_info(t, dt):

    # solving for mrp_r_n_k
    mrp_r_n_k = get_target_mrp_r_n(t)

    # solving for w_r_n
    w_r_n_k = get_target_w_r_n(t)

    # solving for  w_r_n_dot_k
    w_r_n_dot_k = ( get_target_w_r_n(t+dt) - w_r_n_k ) / dt

    return mrp_r_n_k, w_r_n_k, w_r_n_dot_k

def get_w_b_r_k(mrp_b_r_k, w_b_n_k, w_r_n_k):

    b_r_dcm = mrp_functions.mrp_to_dcm(mrp_b_r_k)
    return w_b_n_k - b_r_dcm @ w_r_n_k

def get_mrp_b_r(mrp_b_n_k, mrp_r_n_k):
    """
    Compute the MRP of frame B relative to frame R.

    Args:
        mrp_b_n_k: MRP of B relative to N.
        mrp_r_n_k: MRP of R relative to N.

    Returns:
        mrp_b_r: MRP of B relative to R.
    """
    return mrp_functions.mrp_addition(mrp_b_n_k, mrp_functions.invert_mrp(mrp_r_n_k))

def get_att_track_control_k(x_k, t):
    """
    Compute the control torque for attitude tracking.

    Args:
        x_k: Current state [MRP_B/N, w_B/N].
        t: Current time.

    Returns:
        Control torque vector (u).
    """
    # State decomposition
    mrp_b_n_k = x_k[:3]  # MRP_B/N
    w_b_n_k = x_k[3:6]   # Angular velocity_B/N (in B)

    # Target info
    mrp_r_n_k, w_r_n_k, w_r_n_dot_k = get_target_info(t, dt)  # Target MRP and angular velocity

    # Step 1: Compute relative MRP (sigma_B/R)
    mrp_b_r_k = get_mrp_b_r(mrp_b_n_k, mrp_r_n_k)

    # Step 2: Transform target angular velocity and acceleration to Body frame
    dcm_b_r = mrp_functions.mrp_to_dcm(mrp_b_r_k)  # DCM from R to B
    w_r_n_b = dcm_b_r @ w_r_n_k                   # Transform w_R/N to Body frame
    w_r_n_dot_b = dcm_b_r @ w_r_n_dot_k           # Transform w_R/N_dot to Body frame

    # Step 3: Compute relative angular velocity (omega_B/R in Body frame)
    w_b_r_k = w_b_n_k - w_r_n_b

    # Step 4: Compute control torque
    control_torque = (
        -k * mrp_b_r_k                         # Proportional term (error in attitude)
        - p @ w_b_r_k                          # Damping term (relative angular velocity)
        + inertia_tensor @ (w_r_n_dot_b - np.cross(w_b_n_k, w_r_n_b))  # Inertia term
    )

    return control_torque

def get_att_track_state_dot(x_k, t):

    # spacecraft
    mrp_b_n_k = x_k[:3]
    w_b_n_k = x_k[3:6]

    # target
    mrp_r_n_k, w_r_n_k, _ = get_target_info(t, dt)

    # solving for mrp_b_n_dot_k
    mrp_b_n_norm = np.linalg.norm(mrp_b_n_k)
    mrp_b_n_tilde = helper_functions.get_tilde_matrix(mrp_b_n_k)

    mrp_b_r_k = get_mrp_b_r(mrp_b_n_k, mrp_r_n_k)
    w_b_r_k = get_w_b_r_k(mrp_b_r_k, w_b_n_k, w_r_n_k)

    mrp_b_n_dot = ( 1/4 ) * ( ( 1 - mrp_b_n_norm**2 ) * np.eye(3) + 2 * mrp_b_n_tilde + 2 * np.outer(mrp_b_n_k, mrp_b_n_k) ) @ w_b_r_k

    # solving for w_b_n_dot_k
    w_b_n_dot = np.linalg.inv(inertia_tensor) @ get_att_reg_control_n(x_k) 

    return np.concatenate( [ mrp_b_n_dot, w_b_n_dot ])

def get_track_error_at_time(mrp_sum, time, dt=0.01):

    mrp_b_n_k = mrp_sum[int(time/dt)][:3]
    mrp_r_n_k = get_target_mrp_r_n(time)

    mrp_b_r = get_mrp_b_r(mrp_b_n_k, mrp_r_n_k)
    print(np.linalg.norm(mrp_b_r))


mrp_sum = integrator.runge_kutta(get_att_track_state_dot, x_0, 0, 120, is_mrp=True)
get_track_error_at_time(mrp_sum, 40)