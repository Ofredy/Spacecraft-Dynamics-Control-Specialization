import numpy as np

from hw_functions import integrator


########### problem 1 ###########
def get_beta_p_t(beta_t, t):

    b0, b1, b2, b3 = beta_t[0], beta_t[1], beta_t[2], beta_t[3]

    beta_matrix = np.array([[ -b1, -b2, -b3 ],
                            [ b0, -b3, b2 ],
                            [ b3, b0, -b1 ],
                            [ -b2, b1, b0 ]])

    w_t = np.array([np.sin(0.1*t), 0.01, np.cos(0.1*t)]) * 20 * np.pi / 180

    return (1/2) * np.dot(beta_matrix, w_t)

quaternion_summary = integrator.integrator(get_beta_p_t, np.array([0.408248, 0, 0.408248, 0.816497]), 0, 42, 0.5, norm_value=True)

# printing vector part of time step 42 [s]
b1, b2, b3 = quaternion_summary[84][1], quaternion_summary[84][2], quaternion_summary[84][3]
print(np.sqrt( b1**2 + b2**2 + b3**2 ))