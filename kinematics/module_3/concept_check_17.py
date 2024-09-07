import numpy as np

from hw_functions import mrp_functions


######### problem 1 #########
print(mrp_functions.mrp_is_long(np.array([0.1, 0.2, 0.3])))


######### problem 2 #########
print(mrp_functions.mrp_is_long(np.array([1.2, -0.1, -0.001])))


######### problem 3 #########
print(mrp_functions.get_shadow_mrp(np.array([0.1, 0.2, 0.3])))
