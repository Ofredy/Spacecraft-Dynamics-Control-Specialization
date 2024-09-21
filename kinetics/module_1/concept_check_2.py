# library imports
import numpy as np

# our imports
from hw_functions import dynamic_functions


# Example with 4 particles in 3D space
masses = np.array([1, 1, 2, 2])  # Mass of each particle
velocities = np.array([
    [2, 1, 1],   # Velocity of particle 1
    [0, -1, 1],  # Velocity of particle 2
    [3, 2, -1],  # Velocity of particle 3
    [0, 0, 1]    # Velocity of particle 4
])

print(dynamic_functions.find_total_ke(masses, velocities))
