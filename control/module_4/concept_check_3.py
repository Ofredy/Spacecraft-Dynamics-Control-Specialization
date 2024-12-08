import numpy as np


############### problem 3 ###############

# Define the vectors as numpy arrays
gs1 = np.array([0.267261, 0.534522, 0.801784])
gs2 = np.array([-0.267261, 0.534522, 0.801784])
gs3 = np.array([0.534522, 0.267261, 0.801784])
gs4 = np.array([-0.666667, 0.666667, 0.333333])

# Construct the matrix where each column is one of the vectors
G = np.column_stack((gs1, gs2, gs3, gs4))

print(G.shape)

required_torque = np.array([0.1, 0.2, 0.4])

print( G.T @ np.linalg.inv( G @ G.T ) @ required_torque )
