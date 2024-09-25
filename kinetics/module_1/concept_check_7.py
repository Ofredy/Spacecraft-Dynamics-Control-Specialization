# library imports
import numpy as np

# our imports
from hw_functions import dynamic_functions

########## problem 1 ##########
inertia_tensor_B = np.array([[ 10, 1, -1 ],
                             [ 1, 5, 1 ],
                             [ -1, 1, 8 ]]) 

angular_velocity_B = np.array([0.01, -0.01, 0.01])

print(dynamic_functions.find_rot_ke(inertia_tensor_B, angular_velocity_B.reshape(-1, 1)))
