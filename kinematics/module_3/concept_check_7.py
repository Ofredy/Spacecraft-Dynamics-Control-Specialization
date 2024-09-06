import numpy as np

from hw_functions import quaternions


######## problem 1 ########
q_FB = np.array([0.359211, 0.898027, 0.179605, 0.179605])
q_BN = np.array([0.774597, 0.258199, 0.516398, 0.258199])

print(quaternions.quaternion_addition(q_FB, q_BN))


######## problem 2 ########
q_FN = np.array([0.359211, 0.898027, 0.179605, 0.179605])
q_BN = np.array([-0.377964, 0.755929, 0.377964, 0.377964])

print(quaternions.quaternion_addition(q_FN, quaternions.invert_quaternion(q_BN)))