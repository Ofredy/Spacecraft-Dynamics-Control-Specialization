import numpy as np

from hw_functions import integrator


######### problem 1 #########
def get_q_p_t(q_n, t):

    q1, q2, q3 = q_n[0], q_n[1], q_n[2]

    q_matrix = np.array([[ 1 + q1**2, q1*q2 - q3, q1*q3 + q2 ],
                         [ q2*q1 + q3, 1 + q2**2, q2*q3 - q1 ],
                         [ q3*q1 - q2, q3*q2 + q1, 1 + q3**2 ]])

    w_t = np.array([ np.sin(0.1*t), 0.01, np.cos(0.1*t) ]) * 3 * np.pi / 180

    return (1/2) * np.dot(q_matrix, w_t)

crp_summary = integrator.integrator(get_q_p_t, np.array([0.4, 0.2, -0.1]), 0, 42)
print(np.linalg.norm(crp_summary[420]))