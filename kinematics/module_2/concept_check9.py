import numpy as np
import sympy as sp

from hw_functions import integrator, dcm_functions


############ problem 1 ############
x1, x2, x3 = sp.symbols('x1 x2 x3')

dcm = dcm_functions.get_dcm("2-3-2", x1, x2, x3)
print('symbolic 2-3-2 matrix')
print(dcm)


############ problem 2 ############
w_t = lambda x_n, t: np.array([np.sin(0.1*t), 0.01, np.cos(0.1*t)]) * 20 * np.pi / 180

euler_angle_summary = integrator.integrator(w_t, np.radians(np.array([40, 30, 80])), 0, 60, 0.5)

print(np.linalg.norm(euler_angle_summary[84]))