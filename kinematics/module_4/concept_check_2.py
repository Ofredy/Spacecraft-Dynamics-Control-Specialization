import numpy as np

from hw_functions import attitude_estimation


############ problem 1 ############
print(attitude_estimation.triad_method(np.array([0.8273, 0.5541, -0.0920]), np.array([-0.8285, 0.5522, -0.0955]), 
                                       np.array([-0.1517, -0.9669, 0.2050]), np.array([-0.8393, 0.4494, -0.3044]))) 


true_dcm = np.array([[ 0.963592, 0.187303, 0.190809 ],
                     [ -0.223042, 0.956645, 0.187303 ],
                     [ -0.147454, -0.223042, 0.963592 ]])

estimate_dcm = np.array([[ 0.969846, 0.17101, 0.173648 ],
                         [ -0.200706, 0.96461, 0.17101 ],
                         [ -0.138258, -0.200706, 0.969846 ]])

print(attitude_estimation.estimate_error(true_dcm, estimate_dcm))