import numpy as np
import matplotlib.pyplot as plt

from hw_functions import integrator, helper_functions


################## problem 2 ##################

inertia_tensor = np.array([[ 100.0, 0.0, 0.0 ],
                           [ 0.0, 75.0, 0.0 ],
                           [ 0.0, 0.0, 80.0 ]])

k = 0.11
p = 3 * np.eye(3)

# linear system
def get_linear_control(x_k):
    mrp_k = x_k[:3]
    w_k = x_k[3:]

    mrp_norm = np.linalg.norm(mrp_k)
    w_norm = np.linalg.norm(w_k)

    control_matrix = np.outer(w_k, w_k) + (
        ((4 * k) / (1 + mrp_norm**2)) - w_norm**2 / 2
    ) * np.eye(3)
    
    return -control_matrix @ mrp_k

def get_state_dot(x_k, t):

    mrp_k = x_k[:3]
    w_k = x_k[3:]

    sigma_norm = np.linalg.norm(mrp_k)
    mrp_tilde = helper_functions.get_tilde_matrix(mrp_k)  # Skew-symmetric matrix
    mrp_dot = (1 / 4) * ((1 - sigma_norm**2) * np.eye(3) + 2 * mrp_tilde + 2 * np.outer(mrp_k, mrp_k)) @ w_k

    w_dot = np.linalg.inv(inertia_tensor) @ ( get_linear_control(x_k) - helper_functions.get_tilde_matrix(w_k) @ inertia_tensor @ w_k )

    return np.concatenate([ mrp_dot, w_dot ])

def plot_mrp(mrp_sum, dt=0.01):
    """
    Plot the MRPs over time on a single graph.

    Args:
        mrp_sum: Array of MRPs at each time step (state history).
        dt: Time step size.
    """
    # Convert to NumPy array if it's a list
    mrp_sum = np.array(mrp_sum)

    # Compute the total simulation time from the number of steps
    time_span = (mrp_sum.shape[0] - 1) * dt

    # Time array
    time_array = np.arange(0, time_span + dt, dt)

    # Extract MRPs
    mrps = mrp_sum[:, :3]

    # Create plot
    plt.figure(figsize=(10, 6))
    plt.plot(time_array, mrps[:, 0], label="MRP 1")
    plt.plot(time_array, mrps[:, 1], label="MRP 2")
    plt.plot(time_array, mrps[:, 2], label="MRP 3")
    plt.title("Modified Rodrigues Parameters (MRPs) Over Time")
    plt.xlabel("Time (s)")
    plt.ylabel("MRP")
    plt.grid()
    plt.legend()
    plt.tight_layout()


mrp_0 = np.array([ 0.1, 0.2, -0.1 ])
w_0 = np.array([ 30.0, 10.0, -20.0 ]) * np.pi / 180
x_0 = np.concatenate([ mrp_0, w_0 ])

mrp_sum = integrator.runge_kutta(get_state_dot, x_0, 0, 120, is_mrp=True)
print(np.linalg.norm(mrp_sum[5000][:3]))
plot_mrp(mrp_sum)
plt.show()


################## problem 3 ##################
