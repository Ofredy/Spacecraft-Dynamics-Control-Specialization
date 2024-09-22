# library imports
import numpy as np

# our imports
from hw_functions import dynamic_functions


########### problem 1 ###########
masses = np.array([ 1, 1, 2, 2 ])
velocities = np.array([[ 2, 1, 1 ],
                       [ 0, -1, 1 ],
                       [ 3, 2, -1 ],
                       [ 0, 0, 1 ]])

print(dynamic_functions.find_linear_momentum(masses, velocities))
