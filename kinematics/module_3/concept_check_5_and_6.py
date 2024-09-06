import sys
import math

import numpy as np

from hw_functions import dcm_functions, quaternions


########## problem 1 ##########
print(quaternions.quaternion_to_dcm(np.array([0.235702, 0.471405, -0.471405, 0.707107])))


########## problem 3 ##########
dcm = np.array([[ -0.529403, -0.467056, 0.708231 ],
                [ -0.474115, -0.529403, -0.703525],
                [ 0.703525, -0.708231, 0.0588291 ]])

print(quaternions.dcm_to_quaternions(dcm))


########## problem 4 ##########
x1, x2, x3 = math.radians(20), math.radians(10), math.radians(-10)

dcm = dcm_functions.get_dcm("3-2-1", x1, x2, x3)

print(quaternions.dcm_to_quaternions(np.array(dcm.tolist())))