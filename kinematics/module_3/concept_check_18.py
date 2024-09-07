import numpy as np

from hw_functions import mrp_functions


######## problem 1 ######## 
print(mrp_functions.mrp_to_dcm(np.array([0.1, 0.2, 0.3])))


######## problem 2 ######## 
print(mrp_functions.dcm_to_mrp(np.array([[ 0.763314, 0.0946746, -0.639053 ],
                                         [ -0.568047, -0.372781, -0.733728 ],
                                         [ -0.307692, 0.923077, -0.230769 ]])))
