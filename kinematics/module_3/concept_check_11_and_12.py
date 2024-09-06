import numpy as np

from hw_functions import crps


######### problem 1 #########
print(crps.crp_to_dcm(np.array([0.1, 0.2, 0.3])))


######### problem 2 #########
print(crps.dcm_to_crp( np.array([[ 0.333333, -0.666667, 0.666667],
                                 [ 0.871795, 0.487179, 0.0512821 ],
                                 [ -0.358974, 0.564103, 0.74359 ]]) ))


########## problem 3 #########
print(crps.invert_crp(np.array([0.1, 0.2, 0.3])))


########## problem 4 #########
print(crps.crp_addition(np.array([-0.3, 0.3, 0.1]), crps.invert_crp(np.array([0.1, 0.2, 0.3]))))
