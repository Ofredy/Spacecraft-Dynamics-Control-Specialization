# library imports
import math
import numpy as np

# our imports
from hw_functions import dcm_functions, dynamic_functions


########## problem 1 ##########
x1, x2, x3 = math.radians(-10), math.radians(10), math.radians(5)
dcm = dcm_functions.get_dcm('3-2-1', x1, x2, x3)

inertia_tensor_B = np.array([[ 10, 1, -1 ],
                             [ 1, 5, 1 ],
                             [ -1, 1, 8 ]]) 

inertia_offset_N = np.array([-0.5, 0.5, 0.25])
inertia_offset_B = dcm @ inertia_offset_N

total_mass = 12.5

print(dynamic_functions.parallel_axis_theorem(total_mass, inertia_tensor_B, inertia_offset_B))
