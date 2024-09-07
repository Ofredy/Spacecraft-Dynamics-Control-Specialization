import numpy as np

from hw_functions import mrp_functions


########### problem 1 ########### 
print(mrp_functions.invert_mrp(np.array([0.2, 0.2, -0.1])))


########### problem 2 ########### 
print(mrp_functions.mrp_addition(np.array([-0.1, 0.3, 0.1]), np.array([0.1, 0.2, 0.3])))


########### problem 3 ########### 
print(mrp_functions.mrp_addition(np.array([0.1, 0.2, 0.3]), mrp_functions.invert_mrp(np.array([0.5, 0.3, 0.1]))))
