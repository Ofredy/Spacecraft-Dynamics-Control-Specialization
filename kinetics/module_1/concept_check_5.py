# library imports
import math
import numpy as np

# our imports
from hw_functions import dcm_functions, dynamic_functions


########## problem 4 ##########
x1, x2, x3 = math.radians(-10), math.radians(10), math.radians(5)
dcm = dcm_functions.get_dcm('3-2-1', x1, x2, x3)

inertia_tensor_B = np.array([[ 10, 1, -1 ],
                             [ 1, 5, 1 ],
                             [ -1, 1, 8 ]]) 

angular_velocity_N = np.array([0.01, -0.01, 0.01])
angular_velocity_B = dcm @ angular_velocity_N

print(dynamic_functions.find_rot_angular_momentum(inertia_tensor_B, angular_velocity_B))
