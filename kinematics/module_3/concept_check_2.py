import numpy as np

from hw_functions import prv_functions


dcm = np.array([[ 0.925417, 0.336824, 0.173648 ],
                [ 0.0296956, -0.521281, 0.852869 ],
                [ 0.377786, -0.784102, -0.492404 ]])

e_vec, phi_upper = prv_functions.get_prv_params_from_dcm(dcm)

print("e_vec: %.5f, %.5f, %.5f" % (e_vec[0], e_vec[1], e_vec[2]))
print("phi_upper: %.3f" % (phi_upper))
