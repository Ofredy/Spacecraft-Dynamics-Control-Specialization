import math

import numpy as np

from hw_functions import prv_functions, dcm_functions


############# Problem 2 #############
x1, x2, x3 = math.radians(20), math.radians(-10), math.radians(120)

dcm = dcm_functions.get_dcm("3-2-1", x1, x2, x3)

e_vec, phi_upper = prv_functions.get_prv_params_from_dcm(np.array(dcm.tolist()))

print("e_vec: %.5f, %.5f, %.5f" % (e_vec[0], e_vec[1], e_vec[2]))
print("phi_upper: %.3f" % (phi_upper))


############# Problem 3 #############
fb_dcm = np.array([[ 1, 0, 0 ],
                   [ 0, 0, 1 ],
                   [ 0, -1, 0 ]])

bn_dcm = np.array([[ 1, 0, 0 ],
                   [ 0, 0, 1 ],
                   [ 0, -1, 0 ]])

fn_dcm = fb_dcm @ bn_dcm

print(fn_dcm)

e_vec, phi_upper = prv_functions.get_prv_params_from_dcm(fn_dcm)

print("e_vec: %.5f, %.5f, %.5f" % (e_vec[0], e_vec[1], e_vec[2]))
print("phi_upper: %.3f" % (math.degrees(phi_upper)))
